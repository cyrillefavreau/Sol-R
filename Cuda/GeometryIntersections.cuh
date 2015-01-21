/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once
//#define VOLUME_RENDERING_NORMALS

// Project
#include "VectorUtils.cuh"

// Project
#include "CudaDataTypes.h"
#include "../Consts.h"
#include "TextureMapping.cuh"

__device__ __INLINE__ Vertex project( const Vertex& A, const Vertex& B) 
{
   return B*(dot(A,B)/dot(B,B));
}

/*
________________________________________________________________________________

Compute ray attributes
________________________________________________________________________________
*/
__device__ __INLINE__ void computeRayAttributes(Ray& ray)
{
   ray.inv_direction.x = ray.direction.x!=0.f ? 1.f/ray.direction.x : 1.f;
   ray.inv_direction.y = ray.direction.y!=0.f ? 1.f/ray.direction.y : 1.f;
   ray.inv_direction.z = ray.direction.z!=0.f ? 1.f/ray.direction.z : 1.f;
	ray.signs.x = (ray.inv_direction.x < 0);
	ray.signs.y = (ray.inv_direction.y < 0);
	ray.signs.z = (ray.inv_direction.z < 0);
}

/*
________________________________________________________________________________

Box intersection
________________________________________________________________________________
*/
__device__ __INLINE__ bool boxIntersection( 
	const BoundingBox& box, 
	const Ray&     ray,
	const float&   t0,
	const float&   t1)
{
	float tmin,tmax,tymin, tymax, tzmin, tzmax;

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

Skybox mapping
________________________________________________________________________________
*/
__device__ __INLINE__ float4 skyboxMapping(
   const SceneInfo&    sceneInfo,
   Material*           materials, 
   BitmapBuffer*       textures,
   const Ray&          ray
   ) 
{
   Material& material = materials[sceneInfo.skybox.y];
   float4 result = material.color;
   // solve the equation sphere-ray to find the intersections
   Vertex dir = normalize(ray.direction-ray.origin); 

   float a = 2.f*dot(dir,dir);
   float b = 2.f*dot(ray.origin,dir);
   float c = dot(ray.origin,ray.origin)-(sceneInfo.skybox.x*sceneInfo.skybox.x);
   float d = b*b-2.f*a*c;

   if( d<=0.f || a == 0.f) return result;
   float r = sqrt(d);
   float t1 = (-b-r)/a;
   float t2 = (-b+r)/a;

   if( t1<=EPSILON && t2<=EPSILON ) return result; // both intersections are behind the ray origin

   float t=0.f;
   if( t1<=EPSILON ) 
      t=t2;
   else 
      if( t2<=EPSILON )
         t=t1;
      else
         t=(t1<t2) ? t1 : t2;

   if( t<EPSILON ) return result; // Too close to intersection
   Vertex intersection = normalize(ray.origin+t*dir);

   // Intersection found, no get skybox color

   float U = ((atan2(intersection.x, intersection.z)/PI)+1.f)*.5f;
   float V = (asin(intersection.y)/PI)+.5f;

   int u=int(material.textureMapping.x*U);
   int v=int(material.textureMapping.y*V);

   if( material.textureMapping.x != 0 ) u%=material.textureMapping.x;
   if( material.textureMapping.y != 0 ) v%=material.textureMapping.y;

   if( u>=0 && u<material.textureMapping.x && v>=0 && v<material.textureMapping.y )
   {
      int A=(v*material.textureMapping.x+u)*material.textureMapping.w;
      int B=material.textureMapping.x*material.textureMapping.y*material.textureMapping.w;
      int index=A%B;

      // Diffuse
      int i=material.textureOffset.x+index;
      BitmapBuffer r,g,b;
      r = textures[i  ];
      g = textures[i+1];
      b = textures[i+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }

   return result;
}

/*
________________________________________________________________________________

Ellipsoid intersection
________________________________________________________________________________
*/
__device__ __INLINE__ bool ellipsoidIntersection(
	const SceneInfo& sceneInfo,
   const Primitive& ellipsoid,
	Material*  materials, 
   const Ray& ray, 
   Vertex& intersection,
   Vertex& normal,
	float& shadowIntensity) 
{
	// Shadow intensity
	shadowIntensity = 1.f;

   // solve the equation sphere-ray to find the intersections
	Vertex O_C = ray.origin-ellipsoid.p0;
	Vertex dir = normalize(ray.direction);

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

	normal = normalize(normal);
   return true;
}

/*
________________________________________________________________________________

Sphere intersection
________________________________________________________________________________
*/
__device__ __INLINE__ bool sphereIntersection(
	const SceneInfo& sceneInfo,
	const Primitive& sphere, 
	Material*  materials, 
	const Ray& ray, 
	Vertex&    intersection,
	Vertex&    normal,
	float&     shadowIntensity
	) 
{
   bool back=false;
	// solve the equation sphere-ray to find the intersections
	Vertex O_C = ray.origin-sphere.p0;
	Vertex dir = normalize(ray.direction); 

	float a = 2.f*dot(dir,dir);
	float b = 2.f*dot(O_C,dir);
	float c = dot(O_C,O_C) - (sphere.size.x*sphere.size.x);
	float d = b*b-2.f*a*c;

	if( d<=0.f || a == 0.f) return false;
	float r = sqrt(d);
	float t1 = (-b-r)/a;
	float t2 = (-b+r)/a;

	if( t1<=EPSILON && t2<=EPSILON ) return false; // both intersections are behind the ray origin

	float t=0.f;
	if( t1<=EPSILON ) 
   {
		t = t2;
      back=true;
   }
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
		Vertex newCenter;
      newCenter.x = sphere.p0.x + 0.008f*sphere.size.x*cos(sceneInfo.misc.y + intersection.x );
		newCenter.y = sphere.p0.y + 0.008f*sphere.size.y*sin(sceneInfo.misc.y + intersection.y );
		newCenter.z = sphere.p0.z + 0.008f*sphere.size.z*sin(cos(sceneInfo.misc.y + intersection.z ));
		normal  = intersection - newCenter;
	}
	normal = normalize(normal);
   if(back) normal*=-1.f; 

   // Shadow management
   r = dot(dir,normal);
	shadowIntensity = (materials[sphere.materialId.x].transparency.x != 0.f) ? (1.f-fabs(r)) : 1.f;

	return true;
}

/*
________________________________________________________________________________

Cylinder intersection
ref: http://courses.cms.caltech.edu/cs11/material/advcpp/lab7/index.html
________________________________________________________________________________
*/
__device__ __INLINE__ bool cylinderIntersection( 
	const SceneInfo& sceneInfo,
	const Primitive& cylinder,
	Material*  materials, 
	const Ray& ray,
	Vertex&    intersection,
	Vertex&    normal,
	float&     shadowIntensity)
{
	Vertex O_C = ray.origin-cylinder.p0;
	Vertex dir = ray.direction;
	Vertex n   = crossProduct(dir, cylinder.n1);

	float ln = length(n);

	// Parallel? (?)
	if((ln<EPSILON)&&(ln>-EPSILON)) return false;

	n = normalize(n);

	float d = fabs(dot(O_C,n));
	if (d>cylinder.size.y) return false;

	Vertex O = crossProduct(O_C,cylinder.n1);
	float t = -dot(O, n)/ln;
	if( t<0.f ) return false;

	O = normalize(crossProduct(n,cylinder.n1));
	float s=fabs( sqrtf(cylinder.size.x*cylinder.size.x-d*d) / dot( dir,O ) );

	float t1=t-s;
	float t2=t+s;

	// Calculate intersection point
	intersection = ray.origin+t1*dir;
	Vertex HB1 = intersection-cylinder.p0;
	Vertex HB2 = intersection-cylinder.p1;
	float scale1 = dot(HB1,cylinder.n1);
	float scale2 = dot(HB2,cylinder.n1);
	// Cylinder length
	if( scale1 < EPSILON || scale2 > EPSILON ) 
   {
	   intersection = ray.origin+t2*dir;
      HB1 = intersection-cylinder.p0;
	   HB2 = intersection-cylinder.p1;
	   scale1 = dot(HB1,cylinder.n1);
	   scale2 = dot(HB2,cylinder.n1);
	   // Cylinder length
	   if( scale1 < EPSILON || scale2 > EPSILON ) return false;
   }

   Vertex V = intersection-cylinder.p2;
   normal = V-project(V,cylinder.n1);
	normal = normalize(normal);

   // Shadow management
   dir = normalize(dir);
   float r = dot(dir,normal);
   shadowIntensity = 1.f;
   return true;
}

/*
________________________________________________________________________________

Checkboard intersection
________________________________________________________________________________
*/
__device__ __INLINE__ bool planeIntersection( 
	const SceneInfo& sceneInfo,
	const Primitive& primitive,
	Material*        materials,
	BitmapBuffer*    textures,
	const Ray&       ray, 
	Vertex&          intersection,
	Vertex&          normal,
   float&           shadowIntensity,
	bool             reverse
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
		if( primitive.type.x == ptCamera || materials[primitive.materialId.x].textureIds.x != TEXTURE_NONE )
		{
         Vertex specular = {0.f,0.f,0.f}; // TODO?
         Vertex attributes;
         Vertex advancedAttributes;
			color = cubeMapping(sceneInfo, primitive, materials, textures, intersection, normal, specular, attributes, advancedAttributes );
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
__device__ __INLINE__ bool triangleIntersection( 
    const SceneInfo& sceneInfo,
	const Primitive& triangle, 
	const Ray&       ray,
	Vertex&          intersection,
	Vertex&          normal,
	Vertex&          areas,
	float&           shadowIntensity,
   const bool&      processingShadows)
{
   // Reject rays using the barycentric coordinates of
   // the intersection point with respect to T.
   Vertex E01=triangle.p1-triangle.p0;
   Vertex E03=triangle.p2-triangle.p0;
   Vertex P = crossProduct(ray.direction,E03);
   float det = dot(E01,P);
   
   if (fabs(det) < EPSILON) return false;
   
   Vertex T = ray.origin-triangle.p0;
   float a = dot(T,P)/det;
   if (a < 0.f || a > 1.f) return false;

   Vertex Q = crossProduct(T,E01);
   float b = dot(ray.direction,Q)/det;
   if (b < 0.f || b > 1.f) return false;

   // Reject rays using the barycentric coordinates of
   // the intersection point with respect to T′.
   if ((a+b) > 1.f) 
   {
      Vertex E23 = triangle.p0-triangle.p1;
      Vertex E21 = triangle.p1-triangle.p1;
      Vertex P_ = crossProduct(ray.direction,E21);
      float det_ = dot(E23,P_);
      if(fabs(det_) < EPSILON) return false;
      Vertex T_ = ray.origin-triangle.p2;
      float a_ = dot(T_,P_)/det_;
      if (a_ < 0.f) return false;
      Vertex Q_ = crossProduct(T_,E23);
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
   Vertex v0 = (triangle.p0 - intersection);
   Vertex v1 = (triangle.p1 - intersection);
   Vertex v2 = (triangle.p2 - intersection);
   areas.x = 0.5f*length(crossProduct( v1,v2 ));
   areas.y = 0.5f*length(crossProduct( v0,v2 ));
   areas.z = 0.5f*length(crossProduct( v0,v1 ));
   normal = normalize((triangle.n0*areas.x + triangle.n1*areas.y + triangle.n2*areas.z)/(areas.x+areas.y+areas.z));

   if( sceneInfo.parameters.x==1 )
   {
      // Double Sided triangles
      // Reject triangles with normal opposite to ray.
      Vertex N=normalize(ray.direction);
      if( processingShadows )
      {
         if( dot(N,normal)<=0.f ) return false;
      }
      else
      {
         if( dot(N,normal)>=0.f ) return false;
      }
   }

   Vertex dir = normalize(ray.direction);
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

Intersection Shader
________________________________________________________________________________
*/
__device__ float4 intersectionShader( 
	const SceneInfo& sceneInfo,
	const Primitive& primitive, 
	Material*        materials,
	BitmapBuffer*    textures,
	Vertex&          intersection,
	const Vertex&    areas,
   Vertex&          normal,
   Vertex&          specular,
   Vertex&          attributes,
   Vertex&          advancedAttributes)
{
	float4 colorAtIntersection = materials[primitive.materialId.x].color;
   colorAtIntersection.w = 0.f; // w attribute is used to dtermine light intensity of the material

   if( sceneInfo.parameters.y==1 ) // Extended geometry
   {
	   switch( primitive.type.x ) 
	   {
	   case ptCylinder:
	   case ptEnvironment:
	   case ptSphere:
      case ptEllipsoid:
		   {
			   if(materials[primitive.materialId.x].textureIds.x != TEXTURE_NONE)
			   {
				   colorAtIntersection = sphereUVMapping(primitive, materials, textures, intersection, normal, specular, attributes, advancedAttributes );
			   }
			   break;
		   }
	   case ptCheckboard :
		   {
			   if( materials[primitive.materialId.x].textureIds.x != TEXTURE_NONE ) 
			   {
				   colorAtIntersection = cubeMapping( sceneInfo, primitive, materials, textures, intersection, normal, specular, attributes, advancedAttributes );
			   }
			   else 
			   {
				   int x = sceneInfo.viewDistance.x + ((intersection.x - primitive.p0.x)/primitive.size.x);
				   int z = sceneInfo.viewDistance.x + ((intersection.z - primitive.p0.z)/primitive.size.x);
				   if(x%2==0) 
				   {
					   if (z%2==0) 
					   {
						   colorAtIntersection.x = 1.f-colorAtIntersection.x;
						   colorAtIntersection.y = 1.f-colorAtIntersection.y;
						   colorAtIntersection.z = 1.f-colorAtIntersection.z;
					   }
				   }
				   else 
				   {
					   if (z%2!=0) 
					   {
						   colorAtIntersection.x = 1.f-colorAtIntersection.x;
						   colorAtIntersection.y = 1.f-colorAtIntersection.y;
						   colorAtIntersection.z = 1.f-colorAtIntersection.z;
					   }
				   }
			   }
			   break;
		   }
	   case ptXYPlane:
	   case ptYZPlane:
	   case ptXZPlane:
	   case ptCamera:
		   {
			   if( materials[primitive.materialId.x].textureIds.x != TEXTURE_NONE ) 
			   {
				   colorAtIntersection = cubeMapping( sceneInfo, primitive, materials, textures, intersection, normal, specular, attributes, advancedAttributes );
			   }
			   break;
		   }
	   case ptTriangle:
         {
			   if( materials[primitive.materialId.x].textureIds.x != TEXTURE_NONE ) 
			   {
               colorAtIntersection = triangleUVMapping( sceneInfo, primitive, materials, textures, intersection, areas, normal, specular, attributes, advancedAttributes );
			   }
			   break;
         }
      }
   }
   else
   {
	   if( materials[primitive.materialId.x].textureIds.x != TEXTURE_NONE ) 
	   {
         colorAtIntersection = triangleUVMapping( sceneInfo, primitive, materials, textures, intersection, areas, normal, specular, attributes, advancedAttributes );
	   }
   }
	return colorAtIntersection;
}

/*
________________________________________________________________________________

Intersections with primitives
________________________________________________________________________________
*/
__device__ __INLINE__ bool intersectionWithPrimitives(
	const SceneInfo& sceneInfo,
   const PostProcessingInfo& postProcessingInfo,
	BoundingBox* boundingBoxes, const int& nbActiveBoxes,
	Primitive* primitives, const int& nbActivePrimitives,
	Material* materials, BitmapBuffer* textures,
	const Ray& ray, 
	const int& iteration,
   int&    closestPrimitive, 
	Vertex& closestIntersection,
	Vertex& closestNormal,
   Vertex& closestAreas,
   float4& closestColor,
	float4& colorBox,
   const int currentmaterialId)
{
	bool intersections = false; 
	float minDistance  = sceneInfo.viewDistance.x;//(iteration<2) ? sceneInfo.viewDistance.x : sceneInfo.viewDistance.x/(iteration+1);

	Ray r;
	r.origin    = ray.origin;
	r.direction = ray.direction-ray.origin;
	computeRayAttributes( r );

   Vertex intersection = {0.f,0.f,0.f};
	Vertex normal       = {0.f,0.f,0.f};
	bool i = false;
	float shadowIntensity = 0.f;

   int cptBoxes = 0;
   while(cptBoxes<nbActiveBoxes)
	{
		BoundingBox& box = boundingBoxes[cptBoxes];
      if( boxIntersection(box, r, 0.f, minDistance) )
		{
			// Intersection with Box
         if( sceneInfo.renderBoxes.x!=0 )  // Box 0 is for light emitting objects
         {
            colorBox += materials[box.startIndex.x%NB_MAX_MATERIALS].color/200.f;
         }
         else
         {
			   // Intersection with primitive within boxes
			   for( int cptPrimitives = 0; cptPrimitives<box.nbPrimitives.x; ++cptPrimitives )
			   { 
				   Primitive& primitive = primitives[box.startIndex.x+cptPrimitives];
               Material& material = materials[primitive.materialId.x];
               if( material.attributes.x==0 || (material.attributes.x==1 && currentmaterialId!= primitive.materialId.x)) // !!!! TEST SHALL BE REMOVED TO INCREASE TRANSPARENCY QUALITY !!!
               {
                  Vertex areas = {0.f,0.f,0.f};
                  if(sceneInfo.parameters.y==1)  // Extended geometry
                  {
				         i = false;
				         switch( primitive.type.x )
				         {
				         case ptEnvironment :
                     case ptSphere:
                        {
						         i = sphereIntersection  ( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity ); 
						         break;
					         }
				         case ptCylinder: 
					         {
						         i = cylinderIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity ); 
						         break;
					         }
                     case ptEllipsoid:
                        {
						         i = ellipsoidIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity );
                           break;
                        }
                     case ptTriangle:
                        {
						         i = triangleIntersection( sceneInfo, primitive, r, intersection, normal, areas, shadowIntensity, false ); 
                           break;
                        }
				         default: 
					         {
						         i = planeIntersection   ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, false); 
						         break;
					         }
				         }
                  }
                  else
                  {
						   i = triangleIntersection( sceneInfo, primitive, r, intersection, normal, areas, shadowIntensity, false ); 
                  }

				      float distance = length(intersection-r.origin);
#ifdef BOOLEAN_OPERATOR
                  // TO REMOVE
                  Vertex wholeCenter={0.f,0.f,postProcessingInfo.param2.x};
				      float distanceToCenter = length(intersection-wholeCenter);
                  if( primitive.materialId.x%4!=0 ) 
                     i &= (distanceToCenter>postProcessingInfo.param1.x);
                  // TO REMOVE
#endif // 0

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
         ++cptBoxes;
		}
      else
      {
         cptBoxes += box.indexForNextBox.x;
      }
	}
	return intersections;
}

/*
________________________________________________________________________________

Shadows computation
We do not consider the object from which the ray is launched...
This object cannot shadow itself !

We now have to find the intersection between the considered object and the ray 
which origin is the considered 3D Vertex and which direction is defined by the 
light source center.
.
. * Lamp                     Ray = Origin -> Light Source Center
.  \
.   \##
.   #### object
.    ##
.      \
.       \  Origin
.--------O-------
.
@return 1.f when pixel is in the shades

________________________________________________________________________________
*/
__device__ __INLINE__ float processShadows(
	const SceneInfo& sceneInfo,
	BoundingBox*  boudingBoxes, const int& nbActiveBoxes,
	Primitive*    primitives,
	Material*     materials,
	BitmapBuffer* textures,
	const int&    nbPrimitives, 
	const Vertex& lampCenter, 
	const Vertex& origin, 
	const int&    objectId,
	const int&    iteration,
   float4&       color)
{
	float result = 0.f;
	int cptBoxes = 0;
   color.x = 0.f;
   color.y = 0.f;
   color.z = 0.f;
   Ray r;
	r.origin    = origin;
	r.direction = lampCenter-origin;
	computeRayAttributes( r );
	float minDistance=(iteration<2) ? sceneInfo.viewDistance.x : sceneInfo.viewDistance.x/(iteration+1);

   while( result<(sceneInfo.shadowIntensity.x-sceneInfo.backgroundColor.w) && cptBoxes<nbActiveBoxes )
	{
		BoundingBox& box = boudingBoxes[cptBoxes];
		if( boxIntersection(box, r, 0.05f, minDistance))
		{
			int cptPrimitives = 0;
			while( result<sceneInfo.shadowIntensity.x && cptPrimitives<box.nbPrimitives.x)
			{
				Vertex intersection = {0.f,0.f,0.f};
				Vertex normal       = {0.f,0.f,0.f};
            Vertex areas        = {0.f,0.f,0.f};
				float  shadowIntensity = 0.f;

				Primitive& primitive = primitives[box.startIndex.x+cptPrimitives];
            if( primitive.index.x!=objectId && materials[primitive.materialId.x].attributes.x==0)
				{

					bool hit = false;
               if(sceneInfo.parameters.y==1)
               {
					   switch(primitive.type.x)
					   {
					   case ptSphere   : hit=sphereIntersection   ( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity ); break;
                  case ptEllipsoid: hit=ellipsoidIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity); break;
					   case ptCylinder :	hit=cylinderIntersection ( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity); break;
					   case ptTriangle :	hit=triangleIntersection ( sceneInfo, primitive, r, intersection, normal, areas, shadowIntensity, true ); break;
					   case ptCamera   : hit=false; break;
					   default         : hit=planeIntersection    ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, false ); break;
					   }
               }
               else
               {
                  hit = triangleIntersection( sceneInfo, primitive, r, intersection, normal, areas, shadowIntensity, true );
               }

               if( hit )
					{
						Vertex O_I = intersection-r.origin;
						Vertex O_L = r.direction;
						float l = length(O_I);
						if( l>EPSILON && l<length(O_L) ) // ??? 0.f
						{
                     float ratio = shadowIntensity*sceneInfo.shadowIntensity.x;
                     if( materials[primitive.materialId.x].transparency.x != 0.f )
                     {
                        O_L=normalize(O_L);
                        float a=fabs(dot(O_L,normal));
                        float r = (materials[primitive.materialId.x].transparency.x == 0.f ) ? 1.f : (1.f-/*0.8f**/materials[primitive.materialId.x].transparency.x);
                        ratio *= r*a;
                        // Shadow color
                        color.x  += ratio*(0.1f-0.1f*materials[primitive.materialId.x].color.x);
                        color.y  += ratio*(0.1f-0.1f*materials[primitive.materialId.x].color.y);
                        color.z  += ratio*(0.1f-0.1f*materials[primitive.materialId.x].color.z);
                     }
                     result += ratio;
                  }
					}
				}
				++cptPrimitives;
			}
   		++cptBoxes;
		}
      else
      {
         cptBoxes+=box.indexForNextBox.x;
      }
	}
	result = (result>sceneInfo.shadowIntensity.x) ? sceneInfo.shadowIntensity.x : result;
	result = (result<0.f) ? 0.f : result;
	return result;
}

/*
________________________________________________________________________________

Primitive shader
________________________________________________________________________________
*/
__device__ __INLINE__ float4 primitiveShader(
   const int& index,
	const SceneInfo&   sceneInfo,
	const PostProcessingInfo&   postProcessingInfo,
	BoundingBox* boundingBoxes, const int& nbActiveBoxes, 
	Primitive* primitives, const int& nbActivePrimitives,
	LightInformation* lightInformation, const int& lightInformationSize, const int& nbActiveLamps,
	Material* materials, BitmapBuffer* textures,
	RandomBuffer* randoms,
   const Vertex& origin,
	Vertex&       normal, 
	const int&    objectId, 
	Vertex&       intersection,
   const Vertex& areas,
   float4&       closestColor,
	const int&    iteration,
	float4&       refractionFromColor,
	float&        shadowIntensity,
	float4&       totalBlinn,
   Vertex&       attributes)
{
	Primitive& primitive = primitives[objectId];
   Material& material = materials[primitive.materialId.x];
	float4 lampsColor = { 0.f, 0.f, 0.f, 0.f };

	// Lamp Impact
	shadowIntensity    = 0.f;

   // Bump
   Vertex bumpNormal={0.f,0.f,0.f};
   Vertex advancedAttributes={0.f,0.f,0.f,0.f};

   // Specular
   Vertex specular;
   specular.x=material.specular.x;
   specular.y=material.specular.y;
   specular.z=material.specular.z;

   // Intersection color
   float4 intersectionColor = intersectionShader( sceneInfo, primitive, materials, textures, intersection, areas, bumpNormal, specular, attributes, advancedAttributes );
   normal += bumpNormal;
   normal = normalize(normal);

	if( /*material.innerIllumination.x!=0.f || */ material.attributes.z!=0 )
   {
      // Wireframe returns constant color
		return intersectionColor; 
   }

   if( sceneInfo.graphicsLevel.x>0 )
   {
	   closestColor *= material.innerIllumination.x;
	   for( int cpt=0; cpt<lightInformationSize; ++cpt ) 
	   {
         int cptLamp = (sceneInfo.pathTracingIteration.x>=NB_MAX_ITERATIONS) ? (sceneInfo.pathTracingIteration.x%lightInformationSize) : 0;
         if(lightInformation[cptLamp].attribute.x != primitive.index.x)
		   {
			   Vertex center;
   		   // randomize lamp center
            center = lightInformation[cptLamp].location;

            int t=(index+sceneInfo.misc.y)%(MAX_BITMAP_SIZE-3);
            Material& m=materials[lightInformation[cptLamp].attribute.y];
            if( sceneInfo.pathTracingIteration.x>=NB_MAX_ITERATIONS)
            {
               float a=m.innerIllumination.y*10.f*sceneInfo.pathTracingIteration.x/sceneInfo.maxPathTracingIterations.x;
               center.x += randoms[t  ]*a;
				   center.y += randoms[t+1]*a;
				   center.z += randoms[t+2]*a;
            }

		      Vertex lightRay = center-intersection;
            float lightRayLength=length(lightRay);
            if( lightRayLength<m.innerIllumination.z )
            {
               float4 shadowColor = {0.f,0.f,0.f,0.f};
			      // --------------------------------------------------------------------------------
			      // Lambert
			      // --------------------------------------------------------------------------------
               lightRay = normalize(lightRay);
	            float lambert = material.innerIllumination.x+dot(normal,lightRay);

               if( lambert>0.f && 
                   sceneInfo.graphicsLevel.x>3 && 
                   iteration<4 && // No need to process shadows after 4 generations of rays... cannot be seen anyway.
                   material.innerIllumination.x==0.f ) 
			      {
				      shadowIntensity = processShadows(
					      sceneInfo, boundingBoxes, nbActiveBoxes,
					      primitives, materials, textures, 
					      nbActivePrimitives, center, 
                     intersection, lightInformation[cptLamp].attribute.x, iteration, shadowColor );
			      }

               if( sceneInfo.graphicsLevel.x>0 )
               {
                  float photonEnergy = sqrt(lightRayLength/m.innerIllumination.z);
                  photonEnergy = (photonEnergy>1.f) ? 1.f : photonEnergy;
                  photonEnergy = (photonEnergy<0.f) ? 0.f : photonEnergy;

                  // Transparent materials are lighted on both sides but the amount of light received by the "dark side" 
                  // depends on the transparency rate.
                  lambert *= (lambert<0.f) ? -materials[primitive.materialId.x].transparency.x : 1.f;

                  if( lightInformation[cptLamp].attribute.y != MATERIAL_NONE )
                  {
                     Material& m=materials[lightInformation[cptLamp].attribute.y];
                     lambert *= m.innerIllumination.x; // Lamp illumination
                  }
                  else
                  {
                     lambert *= lightInformation[cptLamp].color.w;
                  }

                  if( material.innerIllumination.w!=0.f )
                  {
                     // Randomize lamp intensity depending on material noise, for more realistic rendering
                     lambert *= (1.f+randoms[t]*material.innerIllumination.w*100.f); 
                  }

			         lambert *= (1.f-shadowIntensity);
                  lambert += sceneInfo.backgroundColor.w;
                  lambert *= (1.f-photonEnergy);

                  // Lighted object, not in the shades
                  lampsColor += lambert*lightInformation[cptLamp].color - shadowColor;

			         if(sceneInfo.graphicsLevel.x>1 && shadowIntensity<sceneInfo.shadowIntensity.x)
			         {
				         // --------------------------------------------------------------------------------
				         // Blinn - Phong
				         // --------------------------------------------------------------------------------
				         Vertex viewRay = normalize(intersection - origin);
				         Vertex blinnDir = lightRay - viewRay;
				         float temp = sqrt(dot(blinnDir,blinnDir));
				         if (temp != 0.f ) 
				         {
					         // Specular reflection
					         blinnDir = (1.f / temp) * blinnDir;

					         float blinnTerm = dot(blinnDir,normal);
					         blinnTerm = ( blinnTerm < 0.f) ? 0.f : blinnTerm;

					         blinnTerm = specular.x * pow(blinnTerm,specular.y);
                        //blinnTerm *= (1.f-material.transparency.x);
                        blinnTerm *= (1.f-photonEnergy);
                        float4 b=lightInformation[cptLamp].color*lightInformation[cptLamp].color.w*blinnTerm;
					         totalBlinn = (length(totalBlinn)>length(b)) ? totalBlinn : b;
                     
                        // Get transparency from specular map
                        totalBlinn.w = specular.z;
				         }
			         }
               }
		      }
         }

         // Light impact on material
		   closestColor += intersectionColor*lampsColor;

         // Ambient occlusion
         if(material.advancedTextureIds.z!=TEXTURE_NONE)
         {
            closestColor*=advancedAttributes.x;
         }
         
         // Saturate color
		   saturateVector(closestColor);

		   refractionFromColor = intersectionColor; // Refraction depending on color;
		   saturateVector( totalBlinn );
	   }
   }
   else
   {
      closestColor = intersectionColor;
   }
	return closestColor;
}

/*
________________________________________________________________________________

Intersections with primitives
________________________________________________________________________________
*/
__device__ __INLINE__ float4 intersectionsWithPrimitives(
   const int& index,
	const SceneInfo& sceneInfo,
	BoundingBox* boundingBoxes, const int& nbActiveBoxes,
	Primitive* primitives, const int& nbActivePrimitives,
	Material* materials, BitmapBuffer* textures,
   LightInformation* lightInformation, const int& lightInformationSize, const int& nbActiveLamps,
   RandomBuffer*    randoms,
   const PostProcessingInfo& postProcessingInfo,
	const Ray& ray)
{
	Ray r;
	r.origin    = ray.origin;
	r.direction = ray.direction-ray.origin;
	computeRayAttributes( r );

   Vertex intersection = ray.origin;
	Vertex normal       = {0.f,0.f,0.f};
	bool i = false;
	float shadowIntensity = 0.f;

   const int MAXDEPTH=10;
   float4 colors[MAXDEPTH];
   for( int i(0); i<MAXDEPTH; ++i)
   {
      colors[i].x=0.f;
      colors[i].y=0.f;
      colors[i].z=0.f;
      colors[i].w=sceneInfo.viewDistance.x;
   }
#ifdef VOLUME_RENDERING_NORMALS
   int normals[MAXDEPTH];
   memset(&normals[0],0,sizeof(bool)*MAXDEPTH);
#endif // VOLUME_RENDERING_NORMALS

   int nbIntersections=0;
   int cptBoxes = 0;
   while(cptBoxes<nbActiveBoxes)
	{
		BoundingBox& box = boundingBoxes[cptBoxes];
      if( boxIntersection(box, r, 0.f, sceneInfo.viewDistance.x) )
		{
			// Intersection with primitive within boxes
			for( int cptPrimitives = 0; cptPrimitives<box.nbPrimitives.x; ++cptPrimitives )
			{ 
				i = false;
				Primitive& primitive = primitives[box.startIndex.x+cptPrimitives];
            Material& material = materials[primitive.materialId.x];
            Vertex areas = {0.f,0.f,0.f};
            if(sceneInfo.parameters.y==1)  // Extended geometry
            {
				   switch( primitive.type.x )
				   {
				   case ptEnvironment :
               case ptSphere    : i = sphereIntersection  ( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity ); break;
				   case ptCylinder  : i = cylinderIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity ); break;
               case ptEllipsoid : i = ellipsoidIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity );break;
               case ptTriangle  : i = triangleIntersection( sceneInfo, primitive, r, intersection, normal, areas, shadowIntensity, false ); break;
				   default          : i = planeIntersection   ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, false); break;
				   }
            }
            else
            {
					i = triangleIntersection( sceneInfo, primitive, r, intersection, normal, areas, shadowIntensity, false ); 
            }
            if(i)
            {
   		      float  dist = length(intersection-r.origin);
               if( dist>postProcessingInfo.param1.x )
               {
                  ++nbIntersections;
                  float4 color=material.color;
#if 0
                  if(sceneInfo.graphicsLevel.x!=0)
                  {
                     color*=(1.f-material.transparency.x);
                     Vertex attributes;
                     attributes.x=material.reflection.x;
                     attributes.y=material.transparency.x;
                     attributes.z=material.refraction.x;
                     attributes.w=material.opacity.x;
                     float4 rBlinn = {0.f,0.f,0.f,0.f};
                     float4 refractionFromColor;
                     float4 closestColor = material.color;
                     shadowIntensity=0.f;
                     color=primitiveShader( 
                        index,
                        sceneInfo, postProcessingInfo,
                        boundingBoxes, nbActiveBoxes, 
                        primitives, nbActivePrimitives, 
                        lightInformation, lightInformationSize, nbActiveLamps,
                        materials, textures, 
                        randoms, r.origin, normal, 
                        box.startIndex.x+cptPrimitives, intersection, areas, closestColor,
                        0, refractionFromColor, shadowIntensity, rBlinn, attributes );
                  }
#endif // 0
                  for( int i(0); i<MAXDEPTH; ++i)
                  {
                     if( dist<colors[i].w )
                     {
                        float a=fabs(dot(normalize(ray.direction-ray.origin),normal));
                        a *= dist/sceneInfo.viewDistance.x;
                        for( int j(MAXDEPTH-1); j>=i; --j)
                        {
                           colors[j+1]=colors[j];
#ifdef VOLUME_RENDERING_NORMALS
                           normals[j+1]=normals[j];
#endif // VOLUME_RENDERING_NORMALS
                        }
                        colors[i] = color*a;
                        colors[i].w = dist;
#ifdef VOLUME_RENDERING_NORMALS
                        normals[i] = (a<0.f) ? -1 : 1;
#endif // VOLUME_RENDERING_NORMALS
                        break;
                     }
                  }
               }
            }
			}
         ++cptBoxes;
		}
      else
      {
         cptBoxes += box.indexForNextBox.x;
      }
	}

#if 0
   float4 color=colors[0]*sceneInfo.backgroundColor.w;
   if( nbIntersections>0 )
   {
      int N=0;
      float D=colors[0].w;
      const int precision=500;
      const float step=sceneInfo.viewDistance.x/float(precision);
      float alpha=1.f/postProcessingInfo.param2.x;
      int c=0;
      for( int i(0); i<precision && c<MAXDEPTH-1; ++i)
      {
         if(D>colors[c].w && N==0)
         {
            color.x+=colors[c].x*alpha;
            color.y+=colors[c].y*alpha;
            color.z+=colors[c].z*alpha;
         }
         D+=step;
         if( D>=colors[c+1].w )
         { 
            ++c;
#ifdef VOLUME_RENDERING_NORMALS
            N+=normals[c];
#endif // VOLUME_RENDERING_NORMALS
         }
      }
      color.w = 0.f;
      normalize(color);
      /*
      if(length(color)<sceneInfo.backgroundColor.w) 
      { 
         color.x=1.f; 
         color.y=1.f; 
         color.z=0.f; 
      }
      */
   }
#else
   float4 color= {0.f,0.f,0.f,0.f}; //colors[0]*(*sceneInfo).backgroundColor.w;
   for( int i=0; i<MAXDEPTH; ++i)
   {
        color += colors[i]*sceneInfo.backgroundColor.w;
   }
#endif // 0
   color.w = colors[0].w;
   return color;
}

   /*
________________________________________________________________________________

Convert Vertex into OpenGL RGB color
________________________________________________________________________________
*/
__device__ __INLINE__ void makeColor(
	const SceneInfo& sceneInfo,
	float4&          color,
	BitmapBuffer*    bitmap,
	int              index)
{
   int mdc_index = index*gColorDepth; 
	color.x = (color.x>1.f) ? 1.f : color.x;
	color.y = (color.y>1.f) ? 1.f : color.y; 
	color.z = (color.z>1.f) ? 1.f : color.z;
	color.x = (color.x<0.f) ? 0.f : color.x;
	color.y = (color.y<0.f) ? 0.f : color.y; 
	color.z = (color.z<0.f) ? 0.f : color.z;

   switch( sceneInfo.misc.x )
   {
      case otOpenGL: 
	   {
		   // OpenGL
		   bitmap[mdc_index  ] = (BitmapBuffer)(color.x*255.f); // Red
      	bitmap[mdc_index+1] = (BitmapBuffer)(color.y*255.f); // Green
		   bitmap[mdc_index+2] = (BitmapBuffer)(color.z*255.f); // Blue
         break;
	   }
      case otDelphi: 
	   {
		   // Delphi
         int y=index/sceneInfo.size.y;
         int x=index%sceneInfo.size.x;
         int i=(y+1)*sceneInfo.size.y-x-1;
         i *= gColorDepth;		   
         bitmap[i  ] = (BitmapBuffer)(color.z*255.f); // Blue
      	bitmap[i+1] = (BitmapBuffer)(color.y*255.f); // Green
		   bitmap[i+2] = (BitmapBuffer)(color.x*255.f); // Red
         break;
	   }
      case otJPEG: 
	   {
		   // JPEG
		   bitmap[mdc_index+2] = (BitmapBuffer)(color.z*255.f); // Blue
      	bitmap[mdc_index+1] = (BitmapBuffer)(color.y*255.f); // Green
		   bitmap[mdc_index  ] = (BitmapBuffer)(color.x*255.f); // Red
         break;
	   }
   }
}
