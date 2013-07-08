/* 
* Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
*/

#pragma once

// Project
#include "VectorUtils.cuh"

// Project
#include "CudaDataTypes.h"
#include "../Consts.h"
#include "TextureMapping.cuh"

/*
________________________________________________________________________________

Compute ray attributes
________________________________________________________________________________
*/
__device__ inline void computeRayAttributes(Ray& ray)
{
	ray.inv_direction.x = 1.f/ray.direction.x;
	ray.inv_direction.y = 1.f/ray.direction.y;
	ray.inv_direction.z = 1.f/ray.direction.z;
	ray.signs.x = (ray.inv_direction.x < 0);
	ray.signs.y = (ray.inv_direction.y < 0);
	ray.signs.z = (ray.inv_direction.z < 0);
}

/*
________________________________________________________________________________

Box intersection
________________________________________________________________________________
*/
__device__ inline bool boxIntersection( 
	const BoundingBox& box, 
	const Ray&     ray,
	const float&   t0,
	const float&   t1 )
{
	float tmin, tmax, tymin, tymax, tzmin, tzmax;

	tmin = (box.parameters[ray.signs.x].x - ray.origin.x) * ray.inv_direction.x;
	tmax = (box.parameters[1-ray.signs.x].x - ray.origin.x) * ray.inv_direction.x;
	tymin = (box.parameters[ray.signs.y].y - ray.origin.y) * ray.inv_direction.y;
	tymax = (box.parameters[1-ray.signs.y].y - ray.origin.y) * ray.inv_direction.y;

	if ( (tmin > tymax) || (tymin > tmax) ) 
		return false;

	if (tymin > tmin) tmin = tymin;
	if (tymax < tmax) tmax = tymax;
	tzmin = (box.parameters[ray.signs.z].z - ray.origin.z) * ray.inv_direction.z;
	tzmax = (box.parameters[1-ray.signs.z].z - ray.origin.z) * ray.inv_direction.z;

	if ( (tmin > tzmax) || (tzmin > tmax) ) 
		return false;

	if (tzmin > tmin) tmin = tzmin;
	if (tzmax < tmax) tmax = tzmax;
   return ( (tmin < t1) && (tmax > t0) );
}

/*
________________________________________________________________________________

Ellipsoid intersection
________________________________________________________________________________
*/
__device__ inline bool ellipsoidIntersection(
	const SceneInfo& sceneInfo,
   const Primitive& ellipsoid,
	Material*  materials, 
   const Ray& ray, 
   float3& intersection,
   float3& normal,
	float& shadowIntensity,
   bool& back) 
{
	// Shadow intensity
	shadowIntensity = 1.f;

   // solve the equation sphere-ray to find the intersections
	float3 O_C = ray.origin-ellipsoid.p0;
	float3 dir = normalize(ray.direction);

   float a = 
        ((dir.x*dir.x)/(ellipsoid.size.x*ellipsoid.size.x))
      + ((dir.y*dir.y)/(ellipsoid.size.y*ellipsoid.size.y))
      + ((dir.z*dir.z)/(ellipsoid.size.z*ellipsoid.size.z));
   float b = 
        ((2.f*O_C.x*dir.x)/(ellipsoid.size.x*ellipsoid.size.x))
      + ((2.f*O_C.y*dir.y)/(ellipsoid.size.y*ellipsoid.size.y))
      + ((2.f*O_C.z*dir.z)/(ellipsoid.size.z*ellipsoid.size.z));
   float c = 
        ((O_C.x*O_C.x)/(ellipsoid.size.x*ellipsoid.size.x))
      + ((O_C.y*O_C.y)/(ellipsoid.size.y*ellipsoid.size.y))
      + ((O_C.z*O_C.z)/(ellipsoid.size.z*ellipsoid.size.z))
      - 1.f;

   float d = ((b*b)-(4.f*a*c));
   if ( d<0.f || a==0.f || b==0.f || c==0.f ) 
   { 
      return false;
   }
   d = sqrt(d); 

   float t1 = (-b+d)/(2.f*a);
   float t2 = (-b-d)/(2.f*a);

	if( t1<=EPSILON && t2<=EPSILON ) return false; // both intersections are behind the ray origin
	back = (t1<=EPSILON || t2<=EPSILON); // If only one intersection (t>0) then we are inside the sphere and the intersection is at the back of the sphere

	float t=0.f;
	if( t1<=EPSILON ) 
		t = t2;
	else 
		if( t2<=EPSILON )
			t = t1;
		else
			t=(t1<t2) ? t1 : t2;

	if( t<EPSILON ) return false; // Too close to intersection
   intersection = ray.origin + t*dir;

   normal = intersection-ellipsoid.p0;
   normal.x = 2.f*normal.x/(ellipsoid.size.x*ellipsoid.size.x);
   normal.y = 2.f*normal.y/(ellipsoid.size.y*ellipsoid.size.y);
   normal.z = 2.f*normal.z/(ellipsoid.size.z*ellipsoid.size.z);

	normal *= (back) ? -1.f : 1.f;
	normal = normalize(normal);
   return true;
}

/*
________________________________________________________________________________

Sphere intersection
________________________________________________________________________________
*/
__device__ inline bool sphereIntersection(
	const SceneInfo& sceneInfo,
	const Primitive& sphere, 
	Material*  materials, 
	const Ray& ray, 
	float3&    intersection,
	float3&    normal,
	float&     shadowIntensity,
	bool&      back
	) 
{
	// solve the equation sphere-ray to find the intersections
	float3 O_C = ray.origin-sphere.p0;
	float3 dir = normalize(ray.direction); 

	float a = 2.f*dot(dir,dir);
	float b = 2.f*dot(O_C,dir);
	float c = dot(O_C,O_C) - (sphere.size.x*sphere.size.x);
	float d = b*b-2.f*a*c;

	if( d<=0.f || a == 0.f) return false;
	float r = sqrt(d);
	float t1 = (-b-r)/a;
	float t2 = (-b+r)/a;

	if( t1<=EPSILON && t2<=EPSILON ) return false; // both intersections are behind the ray origin
	back = (t1<=EPSILON || t2<=EPSILON); // If only one intersection (t>0) then we are inside the sphere and the intersection is at the back of the sphere

	float t=0.f;
	if( t1<=EPSILON ) 
		t = t2;
	else 
		if( t2<=EPSILON )
			t = t1;
		else
			t=(t1<t2) ? t1 : t2;

	if( t<EPSILON ) return false; // Too close to intersection
	intersection = ray.origin+t*dir;

	// TO REMOVE - For Charts only
	//if( intersection.y < sphere.p0.y ) return false;

	if( materials[sphere.materialId.x].attributes.y == 0) 
	{
		// Compute normal vector
		normal = intersection-sphere.p0;
	}
	else
	{
		// Procedural texture
		float3 newCenter;
      newCenter.x = sphere.p0.x + 0.008f*sphere.size.x*cos(sceneInfo.misc.y + intersection.x );
		newCenter.y = sphere.p0.y + 0.008f*sphere.size.y*sin(sceneInfo.misc.y + intersection.y );
		newCenter.z = sphere.p0.z + 0.008f*sphere.size.z*sin(cos(sceneInfo.misc.y + intersection.z ));
		normal  = intersection - newCenter;
	}
	normal *= (back) ? -1.f : 1.f;
	normal = normalize(normal);

   // Shadow management
   r = dot(dir,normal);
	shadowIntensity = (materials[sphere.materialId.x].transparency.x != 0.f) ? (1.f-fabs(r)) : 1.f;

#if EXTENDED_FEATURES
	// Power textures
	if (materials[sphere.materialId.x].textureInfo.y != TEXTURE_NONE && materials[sphere.materialId.x].transparency.x != 0 ) 
	{
		float3 color = sphereUVMapping(sphere, materials, textures, intersection, timer );
		return ((color.x+color.y+color.z) >= sceneInfo.transparentColor.x ); 
	}
#endif // 0

	return true;
}

/*
________________________________________________________________________________

Cylinder intersection
________________________________________________________________________________
*/
__device__ inline bool cylinderIntersection( 
	const SceneInfo& sceneInfo,
	const Primitive& cylinder,
	Material*  materials, 
	const Ray& ray,
	float3&    intersection,
	float3&    normal,
	float&     shadowIntensity,
	bool&      back) 
{
	back = false;
	float3 O_C = ray.origin-cylinder.p0;
	float3 dir = ray.direction;
	float3 n   = crossProduct(dir, cylinder.n1);

	float ln = length(n);

	// Parallel? (?)
	if((ln<EPSILON)&&(ln>-EPSILON))
		return false;

	n = normalize(n);

	float d = fabs(dot(O_C,n));
	if (d>cylinder.size.y) return false;

	float3 O = crossProduct(O_C,cylinder.n1);
	float t = -dot(O, n)/ln;
	O = normalize(crossProduct(n,cylinder.n1));
	float s=fabs( sqrtf(cylinder.size.x*cylinder.size.x-d*d) / dot( dir,O ) );

	float in=t-s;
	float out=t+s;

	if (in<-EPSILON)
	{
		if(out<-EPSILON)
			return false;
		else 
		{
			t=out;
			back = true;
		}
	}
	else
	{
		if(out<-EPSILON)
		{
			t=in;
		}
		else
		{
			if(in<out)
				t=in;
			else
			{
				t=out;
				back = true;
			}

			if( t<0.f ) return false;

			// Calculate intersection point
			intersection = ray.origin+t*dir;

			float3 HB1 = intersection-cylinder.p0;
			float3 HB2 = intersection-cylinder.p1;

			float scale1 = dot(HB1,cylinder.n1);
			float scale2 = dot(HB2,cylinder.n1);

			// Cylinder length
			if( scale1 < EPSILON || scale2 > EPSILON ) return false;

			if( materials[cylinder.materialId.x].attributes.y == 1) 
			{
				// Procedural texture
				float3 newCenter;
				newCenter.x = cylinder.p0.x + 0.01f*cylinder.size.x*cos(sceneInfo.misc.y/100.f+intersection.x);
				newCenter.y = cylinder.p0.y + 0.01f*cylinder.size.y*sin(sceneInfo.misc.y/100.f+intersection.y);
				newCenter.z = cylinder.p0.z + 0.01f*cylinder.size.z*sin(cos(sceneInfo.misc.y/100.f+intersection.z));
				HB1 = intersection - newCenter;
			}

			normal = normalize(HB1-cylinder.n1*scale1);

         // Shadow management
         dir = normalize(dir);
         float r = dot(dir,normal);
         shadowIntensity = (materials[cylinder.materialId.x].transparency.x != 0.f) ? (1.f-fabs(r)) : 1.f;
         return true;
		}
	}
   return false;
}

/*
________________________________________________________________________________

Checkboard intersection
________________________________________________________________________________
*/
__device__ inline bool planeIntersection( 
	const SceneInfo& sceneInfo,
	const Primitive& primitive,
	Material* materials,
	char*     textures,
	const Ray&      ray, 
	float3&   intersection,
	float3&   normal,
	float&    shadowIntensity,
	bool      reverse
	)
{ 
	bool collision = false;

	float reverted = reverse ? -1.f : 1.f;
	normal =  primitive.n0;
	switch( primitive.type.x ) 
	{
	case ptMagicCarpet:
	case ptCheckboard:
		{
			intersection.y = primitive.p0.y;
			float y = ray.origin.y-primitive.p0.y;
			if( reverted*ray.direction.y<0.f && reverted*ray.origin.y>reverted*primitive.p0.y) 
			{
				intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
				intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.z;
			}
			break;
		}
	case ptXZPlane:
		{
			float y = ray.origin.y-primitive.p0.y;
			if( reverted*ray.direction.y<0.f && reverted*ray.origin.y>reverted*primitive.p0.y) 
			{
				intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
				intersection.y = primitive.p0.y;
				intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.z;
				if( materials[primitive.materialId.x].attributes.z == 2 )  // Wireframe
					collision &= wireFrameMapping(intersection.x, intersection.z, materials[primitive.materialId.x].attributes.w, primitive );
			}
			if( !collision && reverted*ray.direction.y>0.f && reverted*ray.origin.y<reverted*primitive.p0.y) 
			{
				normal = -normal;
				intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
				intersection.y = primitive.p0.y;
				intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.z;
				if( materials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
					collision &= wireFrameMapping(intersection.x, intersection.z, materials[primitive.materialId.x].attributes.w, primitive );
			}
			break;
		}
	case ptYZPlane:
		{
			float x = ray.origin.x-primitive.p0.x;
			if( reverted*ray.direction.x<0.f && reverted*ray.origin.x>reverted*primitive.p0.x ) 
			{
				intersection.x = primitive.p0.x;
				intersection.y = ray.origin.y+x*ray.direction.y/-ray.direction.x;
				intersection.z = ray.origin.z+x*ray.direction.z/-ray.direction.x;
				collision = 
               fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.z;
            if( materials[primitive.materialId.x].innerIllumination.x != 0.f )
            {
               // Chessboard like Lights
               collision &= int(fabs(intersection.z))%4000<2000 && int(fabs(intersection.y))%4000<2000;
            }
				if( materials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
					collision &= wireFrameMapping(intersection.y, intersection.z, materials[primitive.materialId.x].attributes.w, primitive );
			}
			if( !collision && reverted*ray.direction.x>0.f && reverted*ray.origin.x<reverted*primitive.p0.x ) 
			{
				normal = -normal;
				intersection.x = primitive.p0.x;
				intersection.y = ray.origin.y+x*ray.direction.y/-ray.direction.x;
				intersection.z = ray.origin.z+x*ray.direction.z/-ray.direction.x;
				collision = 
					fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
					fabs(intersection.z - primitive.p0.z) < primitive.size.z;
            if( materials[primitive.materialId.x].innerIllumination.x != 0.f )
            {
               // Chessboard like Lights
               collision &= int(fabs(intersection.z))%4000<2000 && int(fabs(intersection.y))%4000<2000;
            }
				if( materials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
					collision &= wireFrameMapping(intersection.y, intersection.z, materials[primitive.materialId.x].attributes.w, primitive );
			}
			break;
		}
	case ptXYPlane:
	case ptCamera:
		{
			float z = ray.origin.z-primitive.p0.z;
			if( reverted*ray.direction.z<0.f && reverted*ray.origin.z>reverted*primitive.p0.z) 
			{
				intersection.z = primitive.p0.z;
				intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
				intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.y - primitive.p0.y) < primitive.size.y;
				if( materials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
					collision &= wireFrameMapping(intersection.x, intersection.y, materials[primitive.materialId.x].attributes.w, primitive );
			}
			if( !collision && reverted*ray.direction.z>0.f && reverted*ray.origin.z<reverted*primitive.p0.z )
			{
				normal = -normal;
				intersection.z = primitive.p0.z;
				intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
				intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
				collision = 
					fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
					fabs(intersection.y - primitive.p0.y) < primitive.size.y;
				if( materials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
					collision &= wireFrameMapping(intersection.x, intersection.y, materials[primitive.materialId.x].attributes.w, primitive );
			}
			break;
		}
	}

	if( collision ) 
	{
		// Shadow intensity
		shadowIntensity = 1.f; //sceneInfo.shadowIntensity.x*(1.f-materials[primitive.materialId.x].transparency.x);

		float4 color = materials[primitive.materialId.x].color;
		if( primitive.type.x == ptCamera || materials[primitive.materialId.x].textureMapping.z != TEXTURE_NONE )
		{
			color = cubeMapping(sceneInfo, primitive, materials, textures, intersection );
         shadowIntensity = color.w;
		}

		if( (color.x+color.y+color.z)/3.f >= sceneInfo.transparentColor.x ) 
		{
			collision = false;
		}
	}
	return collision;
}

/*
________________________________________________________________________________

Triangle intersection
________________________________________________________________________________
*/
__device__ inline bool triangleIntersection( 
    const SceneInfo& sceneInfo,
	const Primitive& triangle, 
	Material*        materials,
	const Ray&       ray,
	float3&          intersection,
	float3&          normal,
	float3&          areas,
	float&           shadowIntensity,
	bool&            back )
{
   back = false;
   // Reject rays using the barycentric coordinates of
   // the intersection point with respect to T.
   float3 E01=triangle.p1-triangle.p0;
   float3 E03=triangle.p2-triangle.p0;
   float3 P = crossProduct(ray.direction,E03);
   float det = dot(E01,P);
   
   if (fabs(det) < EPSILON) return false;
   
   float3 T = ray.origin-triangle.p0;
   float a = dot(T,P)/det;
   if (a < 0.f) return false;
   if (a > 1.f) return false;

   float3 Q = crossProduct(T,E01);
   float b = dot(ray.direction,Q)/det;
   if (b < 0.f) return false;
   if (b > 1.f) return false;

   // Reject rays using the barycentric coordinates of
   // the intersection point with respect to T′.
   if ((a+b) > 1.f) 
   {
      float3 E23 = triangle.p0-triangle.p1;
      float3 E21 = triangle.p1-triangle.p1;
      float3 P_ = crossProduct(ray.direction,E21);
      float det_ = dot(E23,P_);
      if(fabs(det_) < EPSILON) return false;
      float3 T_ = ray.origin-triangle.p2;
      float a_ = dot(T_,P_)/det_;
      if (a_ < 0.f) return false;
      float3 Q_ = crossProduct(T_,E23);
      float b_ = dot(ray.direction,Q_)/det_;
      if (b_ < 0.f) return false;
   }

   // Compute the ray parameter of the intersection
   // point.
   float t = dot(E03,Q)/det;
   if (t < 0) return false;

   // Intersection
   intersection = ray.origin + t*ray.direction;

   // Normal
   normal = triangle.n0;
   float3 v0 = triangle.p0 - intersection;
   float3 v1 = triangle.p1 - intersection;
   float3 v2 = triangle.p2 - intersection;
   areas.x = 0.5f*length(crossProduct( v1,v2 ));
   areas.y = 0.5f*length(crossProduct( v0,v2 ));
   areas.z = 0.5f*length(crossProduct( v0,v1 ));
   normal = normalize((triangle.n0*areas.x + triangle.n1*areas.y + triangle.n2*areas.z)/(areas.x+areas.y+areas.z));

   float3 dir = normalize(ray.direction);
   float r = dot(dir,normal);

   if( r>0.f )
   {
      normal *= -1.f;
   }

   // Shadow management
   shadowIntensity = 1.f;
   return true;
}

/*
________________________________________________________________________________

Intersections with primitives
________________________________________________________________________________
*/
__device__ inline bool intersectionWithPrimitives(
	const SceneInfo& sceneInfo,
	BoundingBox* boundingBoxes, const int& nbActiveBoxes,
	Primitive* primitives, const int& nbActivePrimitives,
	Material* materials, char* textures,
	const Ray& ray, 
	const int& iteration,
   int&    closestPrimitive, 
	float3& closestIntersection,
	float3& closestNormal,
   float3& closestAreas,
	float4& colorBox,
	bool&   back,
   const int currentMaterialId)
{
	bool intersections = false; 
	float minDistance  = sceneInfo.viewDistance.x;

	Ray r;
	r.origin    = ray.origin;
	r.direction = ray.direction-ray.origin;
	computeRayAttributes( r );

   float3 intersection = {0.f,0.f,0.f};
	float3 normal       = {0.f,0.f,0.f};
	bool i = false;
	float shadowIntensity = 0.f;

   for( int cptBoxes = 0; cptBoxes<nbActiveBoxes; ++cptBoxes )
	{
		BoundingBox& box = boundingBoxes[cptBoxes];
		if( boxIntersection(box, r, 0.f, sceneInfo.viewDistance.x) )
		{
			// Intersection with Box
			if( sceneInfo.renderBoxes.x != 0 ) 
         {
            colorBox += materials[cptBoxes%NB_MAX_MATERIALS].color/20.f;
         }

			// Intersection with primitive within boxes
			for( int cptPrimitives = 0; cptPrimitives<box.nbPrimitives.x; ++cptPrimitives )
			{ 
				Primitive& primitive = primitives[box.startIndex.x+cptPrimitives];
            Material& material = materials[primitive.materialId.x];
            if( material.attributes.x==0 || (material.attributes.x==1 && currentMaterialId != primitive.materialId.x)) // !!!! TEST SHALL BE REMOVED TO INCREASE TRANSPARENCY QUALITY !!!
            {
               float3 areas = {0.f,0.f,0.f};
				   i = false;
				   switch( primitive.type.x )
				   {
				   case ptEnvironment :
               case ptSphere:
                  {
						   i = sphereIntersection  ( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity, back ); 
						   break;
					   }
				   case ptCylinder: 
					   {
						   i = cylinderIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity, back ); 
						   break;
					   }
               case ptEllipsoid:
                  {
						   i = ellipsoidIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity, back );
                     break;
                  }
               case ptTriangle:
                  {
						   back = false;
						   i = triangleIntersection( sceneInfo, primitive, materials, r, intersection, normal, areas, shadowIntensity, back ); 
                     break;
                  }
				   default: 
					   {
                     back = false;
						   i = planeIntersection   ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, false); 
						   break;
					   }
				   }

				   float distance = length(intersection-r.origin);
				   if( i && distance>EPSILON && distance<minDistance ) 
				   {
					   // Only keep intersection with the closest object
					   minDistance         = distance;
					   closestPrimitive    = box.startIndex.x+cptPrimitives;
					   closestIntersection = intersection;
					   closestNormal       = normal;
                  closestAreas        = areas;
                  intersections       = true;
				   }
            }
			}
		}
	}
	return intersections;
}
