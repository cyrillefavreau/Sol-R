/* 
* Protein Visualizer
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

// System
#include <iostream>

// Cuda
#include <cuda_runtime_api.h>
#if CUDART_VERSION>=5000
#include <helper_cuda.h>
  #ifdef WIN32
    #include <helper_math.h>
  #else
    #include <cutil_math.h>
  #endif // WIN32
#else
#include <cutil_inline.h>
#include <cutil_math.h>
#endif


// Project
#include "CudaDataTypes.h"
#include "../Consts.h"

// Globals
#define M_PI 3.14159265358979323846
#define EPSILON 10.f

// Device arrays
Primitive*   d_primitives;
BoundingBox* d_boundingBoxes; 
int*         d_lamps;
Material*    d_materials;
char*        d_textures;
float*       d_randoms;
float4*      d_postProcessingBuffer;
char*        d_bitmap;
int*         d_primitivesXYIds;

// ________________________________________________________________________________
__device__ inline float vectorLength( const float4& vector )
{
	return sqrt( vector.x*vector.x + vector.y*vector.y + vector.z*vector.z );
}

// ________________________________________________________________________________
__device__ inline void normalizeVector( float4& v )
{
	v /= vectorLength( v );
}

// ________________________________________________________________________________
__device__ inline void saturateVector( float4& v )
{
	v.x = (v.x<0.f) ? 0.f : v.x;
	v.y = (v.y<0.f) ? 0.f : v.y; 
	v.z = (v.z<0.f) ? 0.f : v.z;
	v.w = (v.w<0.f) ? 0.f : v.w;

	v.x = (v.x>1.f) ? 1.f : v.x;
	v.y = (v.y>1.f) ? 1.f : v.y; 
	v.z = (v.z>1.f) ? 1.f : v.z;
	v.w = (v.w>1.f) ? 1.f : v.w;
}

// ________________________________________________________________________________
__device__ inline float dotProduct( const float4& v1, const float4& v2 )
{
	return ( v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

// ________________________________________________________________________________
__device__ inline float4 crossProduct( const float4& b, const float4& c )
{
	float4 a;
	a.x = b.y*c.z - b.z*c.y;
	a.y = b.z*c.x - b.x*c.z;
	a.z = b.x*c.y - b.y*c.x;
	return a;
}


/*
________________________________________________________________________________
incident  : le vecteur normal inverse a la direction d'incidence de la source 
lumineuse
normal    : la normale a l'interface orientee dans le materiau ou se propage le 
rayon incident
reflected : le vecteur normal reflechi
________________________________________________________________________________
*/
__device__ inline void vectorReflection( float4& r, const float4& i, const float4& n )
{
	r = i-2.f*dotProduct(i,n)*n;
}

__device__ float maxValue( const float& a, const float& b )
{
	return ( a>b ) ? a : b;
}

/*
________________________________________________________________________________
incident: le vecteur norm? inverse ? la direction d?incidence de la source 
lumineuse
n1      : index of refraction of original medium
n2      : index of refraction of new medium
________________________________________________________________________________
*/
__device__ inline void vectorRefraction( 
	float4&      refracted, 
	const float4 incident, 
	const float  n1, 
	const float4 normal, 
	const float  n2 )
{
	refracted = incident;
	if(n1!=n2 && n2!=0.f) 
	{
		float r = n1/n2;
		float cosI = dotProduct( incident, normal );
		float cosT2 = 1.f - r*r*(1.f - cosI*cosI);
		refracted = r*incident + (r*cosI-sqrt( fabs(cosT2) ))*normal;
	}
}

/*
________________________________________________________________________________
__v : Vector to rotate
__c : Center of rotations
__a : Angles
________________________________________________________________________________
*/
__device__ inline void vectorRotation( float4& vector, const float4 center, const float4 angles )
{ 
	float4 result = vector; 
	/* X axis */ 
	result.y = vector.y*cos(angles.x) - vector.z*sin(angles.x); 
	result.z = vector.y*sin(angles.x) + vector.z*cos(angles.x); 
	vector = result; 
	result = vector; 
	/* Y axis */ 
	result.z = vector.z*cos(angles.y) - vector.x*sin(angles.y); 
	result.x = vector.z*sin(angles.y) + vector.x*cos(angles.y); 
	vector = result; 
}

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

Convert float4 into OpenGL RGB color
________________________________________________________________________________
*/
__device__ void makeColor(
	const SceneInfo& sceneInfo,
	float4&   color,
	char*     bitmap,
	int       index)
{
   int mdc_index = index*gColorDepth; 
	color.x = (color.x>1.f) ? 1.f : color.x;
	color.y = (color.y>1.f) ? 1.f : color.y; 
	color.z = (color.z>1.f) ? 1.f : color.z;

   switch( sceneInfo.misc.x )
   {
      case otOpenGL: 
	   {
		   // OpenGL
		   bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
      	bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
		   bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
         break;
	   }
      case otDelphi: 
	   {
		   // Delphi
		   bitmap[mdc_index  ] = (char)(color.z*255.f); // Blue
      	bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
		   bitmap[mdc_index+2] = (char)(color.x*255.f); // Red
         break;
	   }
      case otJPEG: 
	   {
         mdc_index = (sceneInfo.width.x*sceneInfo.height.x-index)*gColorDepth; 
		   // JPEG
		   bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
      	bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
		   bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
         break;
	   }
   }
}

__device__ void juliaSet( const SceneInfo& sceneInfo, const float x, const float y, float4& color )
{
   float W = (float)gTextureWidth;
   float H = (float)gTextureHeight;

   //pick some values for the constant c, this determines the shape of the Julia Set
   float cRe = -0.7f + 0.4f*sinf(sceneInfo.misc.y/1500.f);
   float cIm = 0.27015f + 0.4f*cosf(sceneInfo.misc.y/2000.f);

   //calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
   float newRe = 1.5f * (x - W / 2.f) / (0.5f * W);
   float newIm = (y - H / 2.f) / (0.5f * H);
   //i will represent the number of iterations
   int n;
   //start the iteration process
   float  maxIterations = 40.f+sceneInfo.pathTracingIteration.x;
   for(n = 0; n<maxIterations; n++)
   {
         //remember value of previous iteration
         float oldRe = newRe;
         float oldIm = newIm;
         //the actual iteration, the real and imaginary part are calculated
         newRe = oldRe * oldRe - oldIm * oldIm + cRe;
         newIm = 2.f * oldRe * oldIm + cIm;
         //if the point is outside the circle with radius 2: stop
         if((newRe * newRe + newIm * newIm) > 4.f) break;
   }
   //use color model conversion to get rainbow palette, make brightness black if maxIterations reached
   //color.x += newRe/4.f;
   //color.z += newIm/4.f;
   color.x = 1.f-color.x*(n/maxIterations);
   color.y = 1.f-color.y*(n/maxIterations);
   color.z = 1.f-color.z*(n/maxIterations);
}

__device__ void mandelbrotSet( const SceneInfo& sceneInfo, const float x, const float y, float4& color )
{
   float W = (float)gTextureWidth;
   float H = (float)gTextureHeight;

   float  MinRe		= -2.f;
   float  MaxRe		=	1.f;
   float  MinIm		= -1.2f;
   float  MaxIm		=	MinIm + (MaxRe - MinRe) * H/W;
   float  Re_factor	=	(MaxRe - MinRe) / (W - 1.f);
   double Im_factor	=	(MaxIm - MinIm) / (H - 1.f);
   float  maxIterations = 20.f+sceneInfo.pathTracingIteration.x;

   float c_im = MaxIm - y*Im_factor;
   float c_re = MinRe + x*Re_factor;
   float Z_re = c_re;
   float Z_im = c_im;
   bool isInside = true;
   unsigned n;
   for( n = 0; isInside && n < maxIterations; ++n ) 
   {
      float Z_re2 = Z_re*Z_re;
      float Z_im2 = Z_im*Z_im;
      if ( Z_re2+Z_im2>4.f ) 
      {
         isInside = false;
      }
      Z_im = 2.f*Z_re*Z_im+c_im;
      Z_re = Z_re2 - Z_im2+c_re;
   }

   //color.x += Z_re/64.f;
   //color.y += Z_im/64.f;
   color.x = 1.f-color.x*(n/maxIterations);
   color.y = 1.f-color.y*(n/maxIterations);
   color.z = 1.f-color.z*(n/maxIterations);
}

/*
________________________________________________________________________________

Sphere texture Mapping
________________________________________________________________________________
*/
__device__ float4 sphereUVMapping( 
	const Primitive& primitive,
	Material*        materials,
	char*            textures,
	const float4&    intersection)
{
	float4 result = materials[primitive.materialId.x].color;

	float4 d = primitive.p0-intersection;
	normalize(d);
	int u = primitive.size.x / primitive.materialInfo.x * (0.5f - atan2f(d.z, d.x) / 2*M_PI);
	int v = primitive.size.y / primitive.materialInfo.y * (0.5f - 2.f*(asinf(d.y) / 2*M_PI));

	u = u%gTextureWidth;
	v = v%gTextureHeight;
	if( u>=0 && u<gTextureWidth && v>=0 && v<gTextureHeight )
	{
		int index = gTextureOffset+(materials[primitive.materialId.x].textureInfo.y*gTextureWidth*gTextureHeight + v*gTextureWidth+u)*gTextureDepth;
		unsigned char r = textures[index  ];
		unsigned char g = textures[index+1];
		unsigned char b = textures[index+2];
		result.x = r/256.f;
		result.y = g/256.f;
		result.z = b/256.f;
	}
	return result; 
}

/*
________________________________________________________________________________

Cube texture mapping
________________________________________________________________________________
*/
__device__ float4 cubeMapping(
   const SceneInfo& sceneInfo,
	const Primitive& primitive, 
	Material*        materials,
	char*            textures,
	const float4&    intersection)
{
	float4 result = materials[primitive.materialId.x].color;

#ifdef USE_KINECT
	if( primitive.type.x == ptCamera )
	{
		int x = (intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x;
		int y = gKinectVideoHeight - (intersection.y-primitive.p0.y+primitive.size.y)*primitive.materialInfo.y;

		x = (x+gKinectVideoWidth)%gKinectVideoWidth;
		y = (y+gKinectVideoHeight)%gKinectVideoHeight;

		if( x>=0 && x<gKinectVideoWidth && y>=0 && y<gKinectVideoHeight ) 
		{
			int index = (y*gKinectVideoWidth+x)*gKinectVideo;
			unsigned char r = textures[index+2];
			unsigned char g = textures[index+1];
			unsigned char b = textures[index+0];
			result.x = r/256.f;
			result.y = g/256.f;
			result.z = b/256.f;
		}
	}
	else
#endif // USE_KINECT
	{
		int x = ((primitive.type.x == ptCheckboard) || (primitive.type.x == ptXZPlane) || (primitive.type.x == ptXYPlane))  ? 
			gTextureOffset+(intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x :
		gTextureOffset+(intersection.z-primitive.p0.z+primitive.size.z)*primitive.materialInfo.x;

		int y = ((primitive.type.x == ptCheckboard) || (primitive.type.x == ptXZPlane)) ? 
			gTextureOffset+(intersection.z+primitive.p0.z+primitive.size.z)*primitive.materialInfo.y :
		gTextureOffset+(intersection.y-primitive.p0.y+primitive.size.y)*primitive.materialInfo.y;

		x = x%gTextureWidth;
		y = y%gTextureHeight;

		if( x>=0 && x<gTextureWidth && y>=0 && y<gTextureHeight )
		{
         switch( materials[primitive.materialId.x].textureInfo.y )
         {
         case TEXTURE_MANDELBROT: mandelbrotSet( sceneInfo, x, y, result ); break;
         case TEXTURE_JULIA: juliaSet( sceneInfo, x, y, result ); break;
         default:
            {
			      int index = gTextureOffset+(materials[primitive.materialId.x].textureInfo.y*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
			      unsigned char r = textures[index];
			      unsigned char g = textures[index+1];
			      unsigned char b = textures[index+2];
			      result.x = r/256.f;
			      result.y = g/256.f;
			      result.z = b/256.f;
            }
            break;
         }
		}
	}
	return result;
}

__device__ bool wireFrameMapping( float x, float y, int width, const Primitive& primitive )
{
	int X = abs(x);
	int Y = abs(y);
	int A = primitive.materialInfo.x;
	int B = primitive.materialInfo.y;
	return ( X%A<=width ) || ( Y%B<=width );
}

#if 0
/*
________________________________________________________________________________

Magic Carpet texture mapping
________________________________________________________________________________
*/
__device__ float4 magicCarpetMapping( 
	Primitive primitive, 
	Material* materials,
	char*     textures,
	float4    intersection,
	int*      levels,
	float     timer)
{
	float4 result = materials[primitive.materialId.x].color;
	int x = gTextureOffset+(intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x*5.f;
	int y = gTextureOffset+(intersection.z+timer-primitive.p0.z+primitive.size.y)*primitive.materialInfo.y*50.f;

	x = x%gTextureWidth;
	y = y%gTextureHeight;

	if( x>=0 && x<gTextureWidth && y>=0 && y<gTextureHeight )
	{
		// Level management
		int tid_x = (intersection.x-primitive.p0.x+primitive.size.x      )/(primitive.size.x/2.5f);
		int tid_y = (intersection.z-primitive.p0.z+primitive.size.y+timer)/(primitive.size.y/25.f);
		int tid = tid_x+tid_y*5;
		tid = tid%5000;
		int index = (levels[tid]*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
		unsigned char r = textures[index];
		unsigned char g = textures[index+1];
		unsigned char b = textures[index+2];
		result.x = r/256.f;
		result.y = g/256.f;
		result.z = b/256.f;
	}
	return result;
}

/*
________________________________________________________________________________

Magic Cylinder texture mapping
________________________________________________________________________________
*/
__device__ float4 magicCylinderMapping( 
	Primitive primitive, 
	Material* materials,
	char*     textures,
	float4    intersection,
	int*      levels,
	float     timer)
{
	float4 result = materials[primitive.materialId.x].color;

	int x = gTextureOffset+(intersection.x-      primitive.p0.x+primitive.size.x)*primitive.materialInfo.x*5.f;
	int y = gTextureOffset+(intersection.z+timer-primitive.p0.z+primitive.size.y)*primitive.materialInfo.y*50.f;

	x = x%gTextureWidth;
	y = y%gTextureHeight;

	if( x>=0 && x<gTextureWidth && y>=0 && y<gTextureHeight )
	{
		int tid_x = (intersection.x-primitive.p0.x+primitive.size.x      )/(primitive.size.x/2.5f);
		int tid_y = (intersection.z-primitive.p0.z+primitive.size.y+timer)/(primitive.size.y/25.f);
		int tid = tid_x+tid_y*5;
		tid = tid%5000;
		int index = (levels[tid]*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
		unsigned char r = textures[index  ];
		unsigned char g = textures[index+1];
		unsigned char b = textures[index+2];
		result.x = r/256.f;
		result.y = g/256.f;
		result.z = b/256.f;
	}
	return result;
}
#endif // 0

/*
________________________________________________________________________________

Box intersection
________________________________________________________________________________
*/
__device__ inline bool boxIntersection( 
	const BoundingBox& box, 
	const Ray&     ray,
	const float    t0,
	const float    t1)
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
   float4& intersection,
   float4& normal,
	float& shadowIntensity,
   bool& back) 
{
	// Shadow intensity
	shadowIntensity = sceneInfo.shadowIntensity.x*(1.f-materials[ellipsoid.materialId.x].transparency.x);

   // solve the equation sphere-ray to find the intersections
	float4 O_C = ray.origin-ellipsoid.p0;
	float4 dir = ray.direction;
	normalizeVector( dir );

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

	normal.w = 0.f;
	normal *= (back) ? -1.f : 1.f;
	normalizeVector(normal);
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
	char*      textures, 
	const Ray& ray, 
	float4&    intersection,
	float4&    normal,
	float&     shadowIntensity,
	bool&      back
	) 
{
	// solve the equation sphere-ray to find the intersections
	float4 O_C = ray.origin-sphere.p0;
	float4 dir = ray.direction;
	normalizeVector( dir );

	float a = 2.f*dotProduct(dir,dir);
	float b = 2.f*dotProduct(O_C,dir);
	float c = dotProduct(O_C,O_C) - (sphere.size.x*sphere.size.x);
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

	// Shadow intensity
	shadowIntensity = sceneInfo.shadowIntensity.x*(1.f-materials[sphere.materialId.x].transparency.x);

	if( materials[sphere.materialId.x].textureInfo.x == 0) 
	{
		// Compute normal vector
		normal = intersection-sphere.p0;
	}
	else
	{
		// Procedural texture
		float4 newCenter;
      newCenter.x = sphere.p0.x + 0.008f*sphere.size.x*cos(sceneInfo.misc.y + intersection.x );
		newCenter.y = sphere.p0.y + 0.008f*sphere.size.y*sin(sceneInfo.misc.y + intersection.y );
		newCenter.z = sphere.p0.z + 0.008f*sphere.size.z*sin(cos(sceneInfo.misc.y + intersection.z ));
		normal  = intersection - newCenter;
	}
	normal.w = 0.f;
	normal *= (back) ? -1.f : 1.f;
	normalizeVector(normal);

#if EXTENDED_FEATURES
	// Power textures
	if (materials[sphere.materialId.x].textureInfo.y != TEXTURE_NONE && materials[sphere.materialId.x].transparency.x != 0 ) 
	{
		float4 color = sphereUVMapping(sphere, materials, textures, intersection, timer );
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
__device__ bool cylinderIntersection( 
	const SceneInfo& sceneInfo,
	const Primitive& cylinder,
	Material*  materials, 
	char*      textures,
	const Ray& ray,
	float4&    intersection,
	float4&    normal,
	float&     shadowIntensity,
	bool&      back) 
{
	back = false;
	float4 dir = ray.direction;
	/// normalizeVector(dir); // DO NOT NORMALIZE!!!
	float4 RC = ray.origin-cylinder.p0;
	float4 n = crossProduct(dir, cylinder.n1);

	float ln = vectorLength(n);

	// Parallel? (?)
	if((ln<EPSILON)&&(ln>-EPSILON))
		return false;

	normalizeVector(n);

	float d = fabs(dotProduct(RC,n));
	if (d>cylinder.p0.w) return false;

	float4 O = crossProduct(RC,cylinder.n1);
	float t = -dotProduct(O, n)/ln;
	O = crossProduct(n,cylinder.n1);
	normalizeVector(O);
	float s=fabs( sqrtf(cylinder.p0.w*cylinder.p0.w-d*d) / dotProduct( dir,O ) );

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

			float4 HB1 = intersection-cylinder.p0;
			float4 HB2 = intersection-cylinder.p1;

			float scale1 = dotProduct(HB1,cylinder.n1);
			float scale2 = dotProduct(HB2,cylinder.n1);

			// Cylinder length
			if( scale1 < EPSILON || scale2 > EPSILON ) return false;

			if( materials[cylinder.materialId.x].textureInfo.x == 1) 
			{
				// Procedural texture
				float4 newCenter;
				newCenter.x = cylinder.p0.x + 0.01f*cylinder.size.x*cos(sceneInfo.misc.y/100.f+intersection.x);
				newCenter.y = cylinder.p0.y + 0.01f*cylinder.size.y*sin(sceneInfo.misc.y/100.f+intersection.y);
				newCenter.z = cylinder.p0.z + 0.01f*cylinder.size.z*sin(cos(sceneInfo.misc.y/100.f+intersection.z));
				HB1 = intersection - newCenter;
			}

			normal = HB1-cylinder.n1*scale1;
			normal.w = 0.f;

			normalizeVector( normal );

			// Shadow intensity
			shadowIntensity = sceneInfo.shadowIntensity.x*(1.f-materials[cylinder.materialId.x].transparency.x);
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
__device__ bool planeIntersection( 
	const SceneInfo& sceneInfo,
	const Primitive& primitive,
	Material* materials,
	char*     textures,
	const Ray&      ray, 
	float4&   intersection,
	float4&   normal,
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
				if( materials[primitive.materialId.x].textureInfo.z == 1 ) 
					collision &= wireFrameMapping(intersection.x, intersection.z, materials[primitive.materialId.x].textureInfo.w, primitive );
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
				if( materials[primitive.materialId.x].textureInfo.z == 1 ) 
					collision &= wireFrameMapping(intersection.x, intersection.z, materials[primitive.materialId.x].textureInfo.w, primitive );
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
				if( materials[primitive.materialId.x].textureInfo.z == 1 ) 
					collision &= wireFrameMapping(intersection.y, intersection.z, materials[primitive.materialId.x].textureInfo.w, primitive );
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
				if( materials[primitive.materialId.x].textureInfo.z == 1 ) 
					collision &= wireFrameMapping(intersection.y, intersection.z, materials[primitive.materialId.x].textureInfo.w, primitive );
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
				if( materials[primitive.materialId.x].textureInfo.z == 1 ) 
					collision &= wireFrameMapping(intersection.x, intersection.y, materials[primitive.materialId.x].textureInfo.w, primitive );
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
				if( materials[primitive.materialId.x].textureInfo.z == 1 ) 
					collision &= wireFrameMapping(intersection.x, intersection.y, materials[primitive.materialId.x].textureInfo.w, primitive );
			}
			break;
		}
	}

	if( collision ) 
	{
		// Shadow intensity
		shadowIntensity = sceneInfo.shadowIntensity.x*(1.f-materials[primitive.materialId.x].transparency.x);

		float4 color;
		color = materials[primitive.materialId.x].color;
		if( primitive.type.x == ptCamera || materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE )
		{
			color = cubeMapping(sceneInfo, primitive, materials, textures, intersection );
		}

		if( (color.x+color.y+color.z)/3.f >= sceneInfo.transparentColor.x ) 
		{
			collision = false;
		}
		else 
		{
			shadowIntensity = sceneInfo.shadowIntensity.x*
				(1.f-(color.x+color.y+color.z)/3.f*materials[primitive.materialId.x].transparency.x);
		}
	}
	return collision;
}

/*
________________________________________________________________________________

Triangle intersection
________________________________________________________________________________
*/
__device__ bool triangleIntersection( 
   const SceneInfo& sceneInfo,
	const Primitive& triangle, 
	Material* materials,
	const Ray&       ray,
	float4&          intersection,
	float4&          normal,
	float&           shadowIntensity
	) 
{
   // Reject rays using the barycentric coordinates of
   // the intersection point with respect to T.
   float4 E01 = triangle.p1 − triangle.p0;
   float4 E03 = triangle.p2 − triangle.p0;
   float4 P = crossProduct(ray.direction,E03);
   float det = dotProduct(E01,P);
   
   if (fabs(det) < EPSILON) return false;
   
   float4 T = ray.origin − triangle.p0;
   float a = dotProduct(T,P)/det;
   if (a < 0.f) return false;
   if (a > 1.f) return false;

   float4 Q = crossProduct(T,E01);
   float b = dotProduct(ray.direction,Q)/det;
   if (b < 0.f) return false;
   if (b > 1.f) return false;

   // Reject rays using the barycentric coordinates of
   // the intersection point with respect to T′.
   if ((a+b) > 1.f) 
   {
      float4 E23 = triangle.p0 − triangle.p1;
      float4 E21 = triangle.p1 − triangle.p1;
      float4 P_ = crossProduct(ray.direction,E21);
      float det_ = dotProduct(E23,P_);
      if(fabs(det_) < EPSILON) return false;
      float4 T_ = ray.origin − triangle.p2;
      float a_ = dotProduct(T_,P_)/det_;
      if (a_ < 0.f) return false;
      float4 Q_ = crossProduct(T_,E23);
      float b_ = dotProduct(ray.direction,Q_)/det_;
      if (b_ < 0.f) return false;
   }

   // Compute the ray parameter of the intersection
   // point.
   float t = dotProduct(E03,Q)/det;
   if (t < 0) return false;

   intersection = ray.origin + t*ray.direction;
   normal = triangle.n0;
   if( triangle.n0.w == 0.f )
   {
      float4 v0 = triangle.p0 - intersection;
      float4 v1 = triangle.p1 - intersection;
      float4 v2 = triangle.p2 - intersection;
      float a0 = 0.5f*vectorLength(crossProduct( v1,v2 ));
      float a1 = 0.5f*vectorLength(crossProduct( v0,v2 ));
      float a2 = 0.5f*vectorLength(crossProduct( v0,v1 ));
      normal = (triangle.n0*a0 + triangle.n1*a1 + triangle.n2*a2)/(a0+a1+a2);
   }
   normal *= (dotProduct(ray.direction,normal)>0.f) ? -1.f : 1.f;
	shadowIntensity = sceneInfo.shadowIntensity.x*(1.f-materials[triangle.materialId.x].transparency.x);
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
	Material*    materials,
	char*        textures,
	const float4 intersection,
	const bool   back )
{
	float4 colorAtIntersection = materials[primitive.materialId.x].color;
	switch( primitive.type.x ) 
	{
	case ptCylinder:
		{
			if(materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE)
			{
				colorAtIntersection = sphereUVMapping(primitive, materials, textures, intersection );
			}
			break;
		}
	case ptEnvironment:
	case ptSphere:
   case ptEllipsoid:
		{
			if(materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE)
			{
				colorAtIntersection = sphereUVMapping(primitive, materials, textures, intersection );
			}
			break;
		}
	case ptCheckboard :
		{
			if( materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE ) 
			{
				colorAtIntersection = cubeMapping( sceneInfo, primitive, materials, textures, intersection );
			}
			else 
			{
				int x = sceneInfo.viewDistance.x + ((intersection.x - primitive.p0.x)/primitive.p0.w*primitive.materialInfo.x);
				int z = sceneInfo.viewDistance.x + ((intersection.z - primitive.p0.z)/primitive.p0.w*primitive.materialInfo.y);
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
			if( materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE ) 
			{
				colorAtIntersection = cubeMapping( sceneInfo, primitive, materials, textures, intersection );
			}
			break;
		}
#if 0
	case ptTriangle:
		break;
	case ptMagicCarpet:
		{
			if( materials[primitive.materialId.x].textureInfo.y != TEXTURE_NONE ) 
			{
				colorAtIntersection = magicCarpetMapping( primitive, materials, textures, intersection, levels );
			}
			break;
		}
#endif // 0
	}
	return colorAtIntersection;
}

/*
________________________________________________________________________________

Shadows computation
We do not consider the object from which the ray is launched...
This object cannot shadow itself !

We now have to find the intersection between the considered object and the ray 
which origin is the considered 3D float4 and which direction is defined by the 
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
__device__ float processShadows(
	const SceneInfo& sceneInfo,
	BoundingBox*  boudingBoxes, const int& nbActiveBoxes,
	Primitive*    primitives,
	Material*     materials,
	char*         textures,
	const int&    nbPrimitives, 
	const float4& lampCenter, 
	const float4& origin, 
	const int&    objectId,
	const int&    iteration)
{
	float result = 0.f;
	int cptBoxes = 0;
	while( result<=1.f && cptBoxes < nbActiveBoxes )
	{
		Ray r;
		r.origin    = origin;
		r.direction = lampCenter-origin;
		//normalizeVector(r.direction); // TODO???
		computeRayAttributes( r );

		BoundingBox& box = boudingBoxes[cptBoxes];
		if( boxIntersection(box, r, 0.f, sceneInfo.viewDistance.x/iteration))
		{
			int cptPrimitives = 0;
			while( result<sceneInfo.shadowIntensity.x && cptPrimitives<box.nbPrimitives.x)
			{
				float4 intersection = {0.f,0.f,0.f,0.f};
				float4 normal       = {0.f,0.f,0.f,0.f};
				float  shadowIntensity = 0.f;

				if( (box.startIndex.x+cptPrimitives) != objectId )
				{
					Primitive& primitive = primitives[box.startIndex.x+cptPrimitives];

					bool hit = false;
					bool back;
					switch(primitive.type.x)
					{
					case ptSphere: 
						{
							hit = sphereIntersection  ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, back ); 
							break;
						}
               case ptEllipsoid:
                  {
						   hit = ellipsoidIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity, back );
                     break;
                  }
					case ptCylinder:
						{
							hit = cylinderIntersection( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, back ); 
							break;
						}
					case ptTriangle:
						{
							hit = triangleIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity ); 
							break;
						}
					default:
						{
							hit = planeIntersection   ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, false /*true*/ ); 
							break;
						}
					}

					if( hit )
					{
						float4 O_I = intersection-r.origin;
						float4 O_L = r.direction;
						float length = vectorLength(O_I);
						if( length>EPSILON && length<vectorLength(O_L) )
						{
							result += hit ? (shadowIntensity-materials[primitive.materialId.x].innerIllumination.x) : 0.f;
						}
					}
				}
				cptPrimitives++;
			}
		}
		cptBoxes++;
	}
	result = (result>1.f) ? 1.f : result;
	result = (result<0.f) ? 0.f : result;
	return result;
}

/*
________________________________________________________________________________

Primitive shader
________________________________________________________________________________
*/
__device__ float4 primitiveShader(
	const SceneInfo&   sceneInfo,
	const PostProcessingInfo&   postProcessingInfo,
	BoundingBox* boundingBoxes, int nbActiveBoxes,
	Primitive* primitives, const int& nbActivePrimitives,
	int* lamps, const int& nbActiveLamps,
	Material* materials, char* textures,
	float* randoms,
	const float4& origin,
	const float4& normal, 
	const int&    objectId, 
	const float4& intersection, 
	const int&    iteration,
	float4&       refractionFromColor,
	float&        shadowIntensity,
	float4&       totalBlinn)
{
	Primitive primitive = primitives[objectId];
	float4 color = materials[primitive.materialId.x].color;
	//color += materials[primitive.materialId.x].innerIllumination.x;
	//normalizeVector(color);

	float4 lampsColor = { 0.f, 0.f, 0.f, 0.f };

	// Lamp Impact
	float lambert      = 0.f;
	float totalLambert = (materials[primitive.materialId.x].innerIllumination.x != 0.f) ? 0.8f : sceneInfo.backgroundColor.w; // Ambient light
	shadowIntensity    = 0.f;

	if( materials[primitive.materialId.x].textureInfo.z == 1 )
		return color; //TODO? wireframe have constant color

	//if( sceneInfo.pathTracingIteration.x > 0 && materials[primitive.materialId.x].innerIllumination.x != 0.f ) 
	//   return color; 

	if( primitive.type.x == ptEnvironment )
	{
		// Final color
		color = intersectionShader( 
			sceneInfo, primitive, materials, textures, 
			intersection, false );
	}
	else 
	{
		color *= materials[primitive.materialId.x].innerIllumination.x;

		for( int cptLamps=0; cptLamps<nbActiveLamps; cptLamps++ ) 
		{
			if(lamps[cptLamps] != objectId)
			{
				float4 center;
				float4 size;
				switch( primitives[lamps[cptLamps]].type.x )
				{
				case ptCylinder:
					{
						center = (primitives[lamps[cptLamps]].p0 + primitives[lamps[cptLamps]].p1)/ 2.f;
						size.x = primitives[lamps[cptLamps]].size.y; 
						size.y = primitives[lamps[cptLamps]].size.y; 
						size.z = primitives[lamps[cptLamps]].size.y; 
						break;
					}
				default:
					{
						center = primitives[lamps[cptLamps]].p0; 
						size=primitives[lamps[cptLamps]].size; 
						break;
					}
				}

				if( sceneInfo.pathTracingIteration.x > 0 )
				{
					int t = 3*sceneInfo.pathTracingIteration.x + int(10.f*sceneInfo.misc.y)%100;
					// randomize lamp center
#if 0
					center.x += size.x*randoms[t  ]; //*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
					center.y += size.y*randoms[t+1]; //*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
					center.z += size.z*randoms[t+2]; //*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
#else
					center.x += 10.f*size.x*randoms[t  ]*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
					center.y += 10.f*size.y*randoms[t+1]*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
					center.z += 10.f*size.z*randoms[t+2]*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
#endif
				}

				if( sceneInfo.shadowsEnabled.x && materials[primitive.materialId.x].innerIllumination.x == 0.f ) 
				{
					shadowIntensity = processShadows(
						sceneInfo, boundingBoxes, nbActiveBoxes,
						primitives, materials, textures, 
						nbActivePrimitives, center, 
						intersection, lamps[cptLamps], iteration );
				}

				Material& material = materials[primitives[lamps[cptLamps]].materialId.x];
				float4 lightRay = center - intersection;
				normalizeVector(lightRay);

				// --------------------------------------------------------------------------------
				// Lambert
				// --------------------------------------------------------------------------------
				lambert = (postProcessingInfo.type.x==ppe_ambientOcclusion) ? 0.6f : dotProduct(normal,lightRay);
				lambert = (lambert<0.f) ? 0.f : lambert;
				lambert *= (materials[primitive.materialId.x].refraction.x == 0.f) ? material.innerIllumination.x : 1.f;
				lambert *= (1.f-shadowIntensity);
				totalLambert += lambert;

				// Lighted object, not in the shades
				lampsColor += lambert*material.color*material.innerIllumination.x;

				if( /*materials[primitive.materialId.x].innerIllumination.x == 0.f &&*/ shadowIntensity < sceneInfo.shadowIntensity.x )
				{
					// --------------------------------------------------------------------------------
					// Blinn - Phong
					// --------------------------------------------------------------------------------
					float4 viewRay = intersection - origin;
					normalizeVector(viewRay);

					float4 blinnDir = lightRay - viewRay;
					float temp = sqrt(dotProduct(blinnDir,blinnDir));
					if (temp != 0.f ) 
					{
						// Specular reflection
						blinnDir = (1.f / temp) * blinnDir;

						float blinnTerm = dotProduct(blinnDir,normal);
						blinnTerm = ( blinnTerm < 0.f) ? 0.f : blinnTerm;

						blinnTerm = materials[primitive.materialId.x].specular.x * pow(blinnTerm,materials[primitive.materialId.x].specular.y); //*materials[primitive.materialId.x].specular.w
						totalBlinn += material.color * material.innerIllumination.x * blinnTerm;
					}
				}
			}
		}
		// Final color
		float4 intersectionColor = 
			intersectionShader( sceneInfo, primitive, materials, textures,
			intersection, false );

		color += /*totalLambert**/ intersectionColor*lampsColor;
		saturateVector(color);

		refractionFromColor = intersectionColor; // Refraction depending on color;
		saturateVector( totalBlinn );
	}
	return color;
}

/*
________________________________________________________________________________

Intersections with primitives
________________________________________________________________________________
*/
__device__ bool intersectionWithPrimitives(
	const SceneInfo& sceneInfo,
	BoundingBox* boundingBoxes, const int& nbActiveBoxes,
	Primitive* primitives, const int& nbActivePrimitives,
	Material* materials, char* textures,
	const Ray& ray, 
	const int& iteration,
	int&    closestPrimitive, 
	float4& closestIntersection,
	float4& closestNormal,
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

	float4 intersection = {0.f,0.f,0.f,0.f};
	float4 normal       = {0.f,0.f,0.f,0.f};
	bool i = false;
	float shadowIntensity = 0.f;

	for( int cptBoxes = 0; cptBoxes<nbActiveBoxes; ++cptBoxes )
	{
		BoundingBox& box = boundingBoxes[cptBoxes];
		if( boxIntersection(box, r, 0.f, sceneInfo.viewDistance.x/iteration) )
		{
			// Intersection with Box
			if( sceneInfo.renderBoxes.x != 0 ) 
         {
            colorBox += materials[cptBoxes%NB_MAX_MATERIALS].color / 10.f;
         }

			// Intersection with primitive within boxes
			for( int cptPrimitives = 0; cptPrimitives<box.nbPrimitives.x; ++cptPrimitives )
			{ 
				Primitive& primitive = primitives[box.startIndex.x+cptPrimitives];
            Material& material = materials[primitive.materialId.x];
            if( material.fastTransparency.x==0 || (material.fastTransparency.x==1 && currentMaterialId != primitive.materialId.x)) // !!!! TEST SHALL BE REMOVED TO INCREASE TRANSPARENCY QUALITY !!!
            {
				   i = false;
				   switch( primitive.type.x )
				   {
				   case ptEnvironment :
               case ptSphere:
                  {
						   i = sphereIntersection  ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, back ); 
						   break;
					   }
				   case ptCylinder: 
					   {
						   i = cylinderIntersection( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, back ); 
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
						   i = triangleIntersection( sceneInfo, primitive, materials, r, intersection, normal, shadowIntensity ); 
                     break;
                  }
				   default: 
					   {
                     back = false;
						   i = planeIntersection   ( sceneInfo, primitive, materials, textures, r, intersection, normal, shadowIntensity, false); 
						   break;
					   }
				   }

				   float distance = vectorLength( intersection - r.origin ); // <- Pb ici!!
				   if( i && distance>EPSILON && distance<minDistance ) 
				   {
					   // Only keep intersection with the closest object
					   minDistance         = distance;
					   closestPrimitive    = box.startIndex.x+cptPrimitives;
					   closestIntersection = intersection;
					   closestNormal       = normal;
					   intersections       = true;
				   }
            }
			}
		}
	}
	return intersections;
}

/*
________________________________________________________________________________

Calculate the reflected vector                   

^ Normal to object surface (N)  
Reflection (O_R)  |                              
\ |  Eye (O_E)                    
\| /                             
----------------O--------------- Object surface 
closestIntersection                      

============================================================================== 
colours                                                                                    
------------------------------------------------------------------------------ 
We now have to know the colour of this intersection                                        
Color_from_object will compute the amount of light received by the
intersection float4 and  will also compute the shadows. 
The resulted color is stored in result.                     
The first parameter is the closest object to the intersection (following 
the ray). It can  be considered as a light source if its inner light rate 
is > 0.                            
________________________________________________________________________________
*/
__device__ float4 launchRay( 
	BoundingBox* boundingBoxes, const int& nbActiveBoxes,
	Primitive* primitives, const int& nbActivePrimitives,
	int* lamps, const int& nbActiveLamps,
	Material*  materials, char* textures,
	float*           randoms,
	const Ray&       ray, 
	const SceneInfo& sceneInfo,
	const PostProcessingInfo& postProcessingInfo,
	float4&          intersection,
	float&           depthOfField,
	int&             primitiveXYId)
{
	float4 intersectionColor   = {0.f,0.f,0.f,0.f};

	float4 closestIntersection = {0.f,0.f,0.f,0.f};
	float4 firstIntersection   = {0.f,0.f,0.f,0.f};
	float4 normal              = {0.f,0.f,0.f,0.f};
	int    closestPrimitive  = 0;
	bool   carryon           = true;
	Ray    rayOrigin         = ray;
	float  initialRefraction = 1.f;
	int    iteration         = 0;
	float  previousWeight    = 1.f;
	primitiveXYId = -1;

	float4 recursiveColor = { 0.f, 0.f, 0.f, 0.f };
	float4 recursiveBlinn = { 0.f, 0.f, 0.f, 0.f };

	// Variable declarations
	float  shadowIntensity = 0.f;
	float4 refractionFromColor;
	float4 reflectedTarget;
	float4 colorBox = { 0.f, 0.f, 0.f, 0.f };
	bool   back = false;

   int currentMaterialId=-2;

	while( iteration<(sceneInfo.nbRayIterations.x+sceneInfo.pathTracingIteration.x) && carryon ) 
	{
		// If no intersection with lamps detected. Now compute intersection with Primitives
		if( carryon ) 
		{
			carryon = intersectionWithPrimitives(
				sceneInfo,
				boundingBoxes, nbActiveBoxes,
				primitives, nbActivePrimitives,
				materials, textures,
				rayOrigin,
				iteration,  
				closestPrimitive, closestIntersection, 
				normal, colorBox, back, currentMaterialId);
		}

		if( carryon ) 
		{
         currentMaterialId = primitives[closestPrimitive].materialId.x;

			if ( iteration==0 )
			{
				intersectionColor.x = 0.f;
				intersectionColor.y = 0.f;
				intersectionColor.z = 0.f;
				intersectionColor.w = 0.f;

				firstIntersection = closestIntersection;
				primitiveXYId = closestPrimitive;
			}

         float recursiveColorRatio = 1.f;
			float4 rBlinn = {0.f,0.f,0.f,0.f};

			if( shadowIntensity <= 0.9f ) // No reflection/refraction if in shades
			{
				// ----------
				// Refraction
				// ----------

				if( materials[primitives[closestPrimitive].materialId.x].transparency.x != 0.f ) 
				{
					// Replace the normal using the intersection color
					// r,g,b become x,y,z... What the fuck!!
					if( materials[primitives[closestPrimitive].materialId.x].textureInfo.y != TEXTURE_NONE) 
					{
						normal *= (recursiveColor-0.5f);
					}

					// Back of the object? If so, reset refraction to 1.f (air)
					float refraction = back ? 1.f : materials[primitives[closestPrimitive].materialId.x].refraction.x;

					// Actual refraction
					float4 O_E = rayOrigin.origin - closestIntersection;
					normalizeVector(O_E);
					vectorRefraction( rayOrigin.direction, O_E, refraction, normal, initialRefraction );
					reflectedTarget = closestIntersection - rayOrigin.direction;

               recursiveColorRatio = previousWeight*(1.f-materials[primitives[closestPrimitive].materialId.x].transparency.x);
					previousWeight = previousWeight*materials[primitives[closestPrimitive].materialId.x].transparency.x;

               // Prepare next ray
					initialRefraction = refraction;
				}
				else
				{
					// ----------
					// Reflection
					// ----------
					if( materials[primitives[closestPrimitive].materialId.x].reflection.x != 0.f ) 
					{
						float4 O_E = rayOrigin.origin - closestIntersection;
						vectorReflection( rayOrigin.direction, O_E, normal );

						reflectedTarget = closestIntersection - rayOrigin.direction;
						recursiveColorRatio = previousWeight*(1.f-materials[primitives[closestPrimitive].materialId.x].reflection.x);
						previousWeight = previousWeight*materials[primitives[closestPrimitive].materialId.x].reflection.x;
					}
					else 
					{
						recursiveColorRatio *= previousWeight;
						carryon = false;
					}         
				}
			}
			else 
			{
				recursiveColorRatio *= previousWeight;
				carryon = false;
			}

			// Get object color
			recursiveColor = primitiveShader( 
				sceneInfo, postProcessingInfo,
				boundingBoxes, nbActiveBoxes,
			   primitives, nbActivePrimitives, lamps, nbActiveLamps, materials, textures, 
			   randoms,
			   rayOrigin.origin, normal, closestPrimitive, closestIntersection, 
			   iteration, refractionFromColor, shadowIntensity, rBlinn );

			// Contribute to final color
         intersectionColor += recursiveColor*recursiveColorRatio;
 			recursiveBlinn += rBlinn;
			intersectionColor -= colorBox;

         rayOrigin.origin    = closestIntersection + reflectedTarget*0.00001f; 
			rayOrigin.direction = reflectedTarget;


			// Noise management
			if( sceneInfo.pathTracingIteration.x != 0 && materials[primitives[closestPrimitive].materialId.x].color.w != 0.f)
			{
				// Randomize view
				int rindex = 3.f*sceneInfo.misc.y + sceneInfo.pathTracingIteration.x;
				rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x);
				rayOrigin.direction.x += randoms[rindex  ]*materials[primitives[closestPrimitive].materialId.x].color.w;
				rayOrigin.direction.y += randoms[rindex+1]*materials[primitives[closestPrimitive].materialId.x].color.w;
				rayOrigin.direction.z += randoms[rindex+2]*materials[primitives[closestPrimitive].materialId.x].color.w;
			}
		}
		else
		{
			intersectionColor += previousWeight*sceneInfo.backgroundColor;
		}
		iteration++; 
	}

	intersectionColor += recursiveBlinn;

	saturateVector( intersectionColor );
	intersection = closestIntersection;

	float len = vectorLength(firstIntersection - ray.origin);
   float halfDistance = sceneInfo.viewDistance.x*0.75f;
   if( sceneInfo.misc.z==1 && len>halfDistance)
   {
	   // --------------------------------------------------
	   // Attenation effect (Fog)
	   // --------------------------------------------------
	   len = 1.f-((len-halfDistance)/(sceneInfo.viewDistance.x-halfDistance));
	   len = (len>0.f) ? len : 0.f; 
	   len = (len<1.f) ? len : 1.f; 
	   intersectionColor.x *= len;
	   intersectionColor.y *= len;
	   intersectionColor.z *= len;
	   
      intersectionColor.x += sceneInfo.backgroundColor.x*(1.f-len);
	   intersectionColor.y += sceneInfo.backgroundColor.y*(1.f-len);
	   intersectionColor.z += sceneInfo.backgroundColor.z*(1.f-len);
   }

	// Depth of field
   depthOfField = (len-depthOfField)/sceneInfo.viewDistance.x;
	return intersectionColor;
}


/*
________________________________________________________________________________

Standard renderer
________________________________________________________________________________
*/
__global__ void k_standardRenderer(
	BoundingBox* BoundingBoxes, int nbActiveBoxes,
	Primitive* primitives, int nbActivePrimitives,
	int* lamps, int nbActiveLamps,
	Material*    materials,
	char*        textures,
	float*       randoms,
	float4       origin,
	float4       direction,
	float4       angles,
	SceneInfo    sceneInfo,
	PostProcessingInfo postProcessingInfo,
	float4*      postProcessingBuffer,
	int*         primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

	Ray ray;
	ray.origin = origin;
	ray.direction = direction;

	float4 rotationCenter = {0.f,0.f,0.f,0.f};

	if( sceneInfo.pathTracingIteration.x == 0 )
   {
		postProcessingBuffer[index].x = 0.f;
		postProcessingBuffer[index].y = 0.f;
		postProcessingBuffer[index].z = 0.f;
		postProcessingBuffer[index].w = 0.f;
   }
   else
	{
		// Randomize view
		int rindex = index + sceneInfo.pathTracingIteration.x;
		rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x);
		ray.direction.x += randoms[rindex  ]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		ray.direction.y += randoms[rindex+1]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		ray.direction.z += randoms[rindex+2]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
	}

	float dof = postProcessingInfo.param1.x;
	float4 intersection;


   if( sceneInfo.misc.w == 1 ) // Isometric 3D
   {
      //ray.origin.z = (ray.origin.z<0.f ) ? ray.origin.z : 0.f;
      ray.direction.x = /* ray.direction.x - (ray.origin.z*0.01f)* */ ray.origin.z*0.001f*(float)(x - (sceneInfo.width.x/2));
	   ray.direction.y = /* ray.direction.y - (ray.origin.z*0.01f)* */ -ray.origin.z*0.001f*(float)(y - (sceneInfo.height.x/2));
      //ray.direction.z = 3000.f;
	   ray.origin.x = ray.direction.x;
	   ray.origin.y = ray.direction.y;
      //ray.origin.z = -5000.f;
   }
   else
   {
      float ratio=(float)sceneInfo.width.x/(float)sceneInfo.height.x;
      float2 step;
      step.x=ratio*6400.f/(float)sceneInfo.width.x;
      step.y=6400.f/(float)sceneInfo.height.x;
      ray.direction.x = ray.direction.x - step.x*(float)(x - (sceneInfo.width.x/2));
      ray.direction.y = ray.direction.y + step.y*(float)(y - (sceneInfo.height.x/2));
   }


	vectorRotation( ray.origin, rotationCenter, angles );
	vectorRotation( ray.direction, rotationCenter, angles );

   __shared__ BoundingBox shBoundingBoxes[128];
   if( threadIdx.x==0 && threadIdx.y==0)
   {
      memcpy( shBoundingBoxes, BoundingBoxes, sizeof(BoundingBox)*nbActiveBoxes);
   }
   __syncthreads();

	float4 color = launchRay(
		shBoundingBoxes, nbActiveBoxes,
		primitives, nbActivePrimitives,
		lamps, nbActiveLamps,
		materials, textures, 
		randoms,
		ray, 
		sceneInfo, postProcessingInfo,
		intersection,
		dof,
		primitiveXYIds[index]);

	if( sceneInfo.pathTracingIteration.x == 0 )
	{
		postProcessingBuffer[index].w = dof;
	}
   postProcessingBuffer[index].x += color.x;
   postProcessingBuffer[index].y += color.y;
   postProcessingBuffer[index].z += color.z;
}

/*
________________________________________________________________________________

Anaglyph Renderer
________________________________________________________________________________
*/
__global__ void k_anaglyphRenderer(
	BoundingBox* BoundingBoxes, int nbActiveBoxes,
	Primitive* primitives, int nbActivePrimitives,
	int* lamps, int nbActiveLamps,
	Material*    materials,
	char*        textures,
	float*       randoms,
	float4       origin,
	float4       direction,
	float4       angles,
	SceneInfo    sceneInfo,
	PostProcessingInfo postProcessingInfo,
	float4*      postProcessingBuffer,
	int*         primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

	float4 rotationCenter = {0.f,0.f,0.f,0.f};

	if( sceneInfo.pathTracingIteration.x == 0 )
	{
		postProcessingBuffer[index].x = 0.f;
		postProcessingBuffer[index].y = 0.f;
		postProcessingBuffer[index].z = 0.f;
		postProcessingBuffer[index].w = 0.f;
	}

	float dof = postProcessingInfo.param1.x;
	float4 intersection;
	Ray eyeRay;

	// Left eye
	eyeRay.origin.x = origin.x + sceneInfo.width3DVision.x;
	eyeRay.origin.y = origin.y;
	eyeRay.origin.z = origin.z;

	eyeRay.direction.x = direction.x - 8.f*(float)(x - (sceneInfo.width.x/2));
	eyeRay.direction.y = direction.y + 8.f*(float)(y - (sceneInfo.height.x/2));
	eyeRay.direction.z = direction.z;

	vectorRotation( eyeRay.origin, rotationCenter, angles );
	vectorRotation( eyeRay.direction, rotationCenter, angles );

	float4 colorLeft = launchRay(
		BoundingBoxes, nbActiveBoxes,
		primitives, nbActivePrimitives,
		lamps, nbActiveLamps,
		materials, textures, 
		randoms,
		eyeRay, 
		sceneInfo, postProcessingInfo,
		intersection,
		dof,
		primitiveXYIds[index]);

	// Right eye
	eyeRay.origin.x = origin.x - sceneInfo.width3DVision.x;
	eyeRay.origin.y = origin.y;
	eyeRay.origin.z = origin.z;

	eyeRay.direction.x = direction.x - 8.f*(float)(x - (sceneInfo.width.x/2));
	eyeRay.direction.y = direction.y + 8.f*(float)(y - (sceneInfo.height.x/2));
	eyeRay.direction.z = direction.z;

	vectorRotation( eyeRay.origin, rotationCenter, angles );
	vectorRotation( eyeRay.direction, rotationCenter, angles );
	float4 colorRight = launchRay(
		BoundingBoxes, nbActiveBoxes,
		primitives, nbActivePrimitives,
		lamps, nbActiveLamps,
		materials, textures, 
		randoms,
		eyeRay, 
		sceneInfo, postProcessingInfo,
		intersection,
		dof,
		primitiveXYIds[index]);

	float r1 = colorLeft.x*0.299f + colorLeft.y*0.587f + colorLeft.z*0.114f;
	float b1 = 0.f;
	float g1 = 0.f;

	float r2 = 0.f;
	float g2 = colorRight.y;
	float b2 = colorRight.z;

	postProcessingBuffer[index].x += r1+r2;
	postProcessingBuffer[index].y += g1+g2;
	postProcessingBuffer[index].z += b1+b2;
	if( sceneInfo.pathTracingIteration.x == 0 ) postProcessingBuffer[index].w = dof;
}

/*
________________________________________________________________________________

3D Vision Renderer
________________________________________________________________________________
*/
__global__ void k_3DVisionRenderer(
	BoundingBox* BoundingBoxes, int nbActiveBoxes,
	Primitive*   primitives,    int nbActivePrimitives,
	int* lamps, int nbActiveLamps,
	Material*    materials,
	char*        textures,
	float*       randoms,
	float4       origin,
	float4       direction,
	float4       angles,
	SceneInfo    sceneInfo,
	PostProcessingInfo postProcessingInfo,
	float4*      postProcessingBuffer,
	int*         primitiveXYIds)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

	float4 rotationCenter = {0.f,0.f,0.f,0.f};

	if( sceneInfo.pathTracingIteration.x == 0 )
	{
		postProcessingBuffer[index].x = 0.f;
		postProcessingBuffer[index].y = 0.f;
		postProcessingBuffer[index].z = 0.f;
		postProcessingBuffer[index].w = 0.f;
	}

	float dof = postProcessingInfo.param1.x;
	float4 intersection;
	int halfWidth  = sceneInfo.width.x/2;

	Ray eyeRay;
	if( x<halfWidth ) 
	{
		// Left eye
		eyeRay.origin.x = origin.x + sceneInfo.width3DVision.x;
		eyeRay.origin.y = origin.y;
		eyeRay.origin.z = origin.z;

		eyeRay.direction.x = direction.x - 8.f*(float)(x - (sceneInfo.width.x/2) + halfWidth/2 );
		eyeRay.direction.y = direction.y + 8.f*(float)(y - (sceneInfo.height.x/2));
		eyeRay.direction.z = direction.z;
	}
	else
	{
		// Right eye
		eyeRay.origin.x = origin.x - sceneInfo.width3DVision.x;
		eyeRay.origin.y = origin.y;
		eyeRay.origin.z = origin.z;

		eyeRay.direction.x = direction.x - 8.f*(float)(x - (sceneInfo.width.x/2) - halfWidth/2);
		eyeRay.direction.y = direction.y + 8.f*(float)(y - (sceneInfo.height.x/2));
		eyeRay.direction.z = direction.z;
	}

	vectorRotation( eyeRay.origin, rotationCenter, angles );
	vectorRotation( eyeRay.direction, rotationCenter, angles );

	float4 color = launchRay(
		BoundingBoxes, nbActiveBoxes,
		primitives, nbActivePrimitives,
		lamps, nbActiveLamps,
		materials, textures, 
		randoms,
		eyeRay, 
		sceneInfo, postProcessingInfo,
		intersection,
		dof,
		primitiveXYIds[index]);

	postProcessingBuffer[index].x += color.x;
	postProcessingBuffer[index].y += color.y;
	postProcessingBuffer[index].z += color.z;
	if( sceneInfo.pathTracingIteration.x == 0 ) postProcessingBuffer[index].w = dof;
}


/*
________________________________________________________________________________

Post Processing Effect: Depth of field
________________________________________________________________________________
*/
__global__ void k_depthOfField(
	SceneInfo        sceneInfo,
	PostProcessingInfo PostProcessingInfo,
	float4*          postProcessingBuffer,
	float*           randoms,
	char*            bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;
	float  depth = PostProcessingInfo.param2.x*postProcessingBuffer[index].w;
	int    wh = sceneInfo.width.x*sceneInfo.height.x;

	float4 localColor;
	localColor.x = 0.f;
	localColor.y = 0.f;
	localColor.z = 0.f;

	for( int i=0; i<PostProcessingInfo.param3.x; ++i )
	{
		int ix = i%wh;
		int iy = (i+sceneInfo.width.x)%wh;
		int xx = x+depth*randoms[ix]*0.1f;
		int yy = y+depth*randoms[iy]*0.1f;
		if( xx>=0 && xx<sceneInfo.width.x && yy>=0 && yy<sceneInfo.height.x )
		{
			int localIndex = yy*sceneInfo.width.x+xx;
			if( localIndex>=0 && localIndex<wh )
			{
				localColor += postProcessingBuffer[localIndex];
			}
		}
		else
		{
			localColor += postProcessingBuffer[index];
		}
	}
	localColor /= PostProcessingInfo.param3.x;
	localColor /= (sceneInfo.pathTracingIteration.x+1);
	localColor.w = 1.f;

	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Ambiant Occlusion
________________________________________________________________________________
*/
__global__ void k_ambiantOcclusion(
	SceneInfo        sceneInfo,
	PostProcessingInfo postProcessingInfo,
	float4*          postProcessingBuffer,
	float*           randoms,
	char*            bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;
	float occ = 0.f;
	float4 localColor = postProcessingBuffer[index];
	float  depth = localColor.w;

   const int step = 16;
	for( int X=-step; X<step; X+=2 )
	{
		for( int Y=-step; Y<step; Y+=2 )
		{
			int xx = x+X;
			int yy = y+Y;
			if( xx>=0 && xx<sceneInfo.width.x && yy>=0 && yy<sceneInfo.height.x )
			{
				int localIndex = yy*sceneInfo.width.x+xx;
				if( postProcessingBuffer[localIndex].w>=depth)
				{
					occ += 1.f;
				}
			}
			else
				occ += 1.f;
		}
	}
	//occ /= float((2*step)*(2*step));
	occ /= float(step*step);
	occ += 0.3f; // Ambient light
	localColor.x *= occ;
	localColor.y *= occ;
	localColor.z *= occ;
	localColor /= (sceneInfo.pathTracingIteration.x+1);
	saturateVector( localColor );
	localColor.w = 1.f;

	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Cartoon
________________________________________________________________________________
*/
__global__ void k_cartoon(
	SceneInfo        sceneInfo,
	PostProcessingInfo PostProcessingInfo,
	float4*          postProcessingBuffer,
	float*           randoms,
	char*            bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;
	float4 localColor = postProcessingBuffer[index];

	int r = localColor.x*255/PostProcessingInfo.param3.x;
	int g = localColor.y*255/PostProcessingInfo.param3.x;
	int b = localColor.z*255/PostProcessingInfo.param3.x;

	localColor.x = float(r*PostProcessingInfo.param3.x/255.f);
	localColor.y = float(g*PostProcessingInfo.param3.x/255.f);
	localColor.z = float(b*PostProcessingInfo.param3.x/255.f);
	localColor /= (sceneInfo.pathTracingIteration.x+1);

	localColor.w = 1.f;
	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Ambiant Occlusion
________________________________________________________________________________
*/
__global__ void k_antiAliasing(
	SceneInfo        sceneInfo,
	PostProcessingInfo PostProcessingInfo,
	float4*          postProcessingBuffer,
	float*           randoms,
	char*            bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;
	float4 localColor = {0.f,0.f,0.f,0.f};

	for( int X=-1; X<=1; X+=2 )
	{
		for( int Y=-1; Y<=1; Y+=2 )
		{
			int xx = x+X;
			int yy = y+Y;
			if( xx>=0 && xx<sceneInfo.width.x && yy>=0 && yy<sceneInfo.height.x )
			{
				int localIndex = yy*sceneInfo.width.x+xx;
				localColor += 0.2f*postProcessingBuffer[localIndex];
			}
		}
	}
	localColor /= 4.f;
	localColor += postProcessingBuffer[index];
	localColor /= (sceneInfo.pathTracingIteration.x+1);
	saturateVector( localColor );
	localColor.w = 1.f;

	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Default
________________________________________________________________________________
*/
__global__ void k_default(
	SceneInfo        sceneInfo,
	PostProcessingInfo PostProcessingInfo,
	float4*          postProcessingBuffer,
	char*            bitmap) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int index = y*sceneInfo.width.x+x;

   float4 localColor = postProcessingBuffer[index]/(float)(sceneInfo.pathTracingIteration.x+1.f);

	makeColor( sceneInfo, localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

GPU initialization
________________________________________________________________________________
*/
extern "C" void initialize_scene( 
	int width, int height, int nbPrimitives, int nbLamps, int nbMaterials, int nbTextures )
{
	// Scene resources
	checkCudaErrors(cudaMalloc( (void**)&d_boundingBoxes,      NB_MAX_BOXES*sizeof(BoundingBox)));
	checkCudaErrors(cudaMalloc( (void**)&d_primitives,         NB_MAX_PRIMITIVES*sizeof(Primitive)));
	checkCudaErrors(cudaMalloc( (void**)&d_lamps,              NB_MAX_LAMPS*sizeof(int)));
	checkCudaErrors(cudaMalloc( (void**)&d_materials,          NB_MAX_MATERIALS*sizeof(Material)));
	checkCudaErrors(cudaMalloc( (void**)&d_textures,           NB_MAX_TEXTURES*gTextureDepth*gTextureWidth*gTextureHeight + gTextureOffset));
	checkCudaErrors(cudaMalloc( (void**)&d_randoms,            width*height*sizeof(float)));

	// Rendering canvas
	checkCudaErrors(cudaMalloc( (void**)&d_postProcessingBuffer,  width*height*sizeof(float4)));
	checkCudaErrors(cudaMalloc( (void**)&d_bitmap,                width*height*gColorDepth*sizeof(char)));
	checkCudaErrors(cudaMalloc( (void**)&d_primitivesXYIds,       width*height*gColorDepth*sizeof(int)));

#if 0
	std::cout <<"GPU: SceneInfo         : " << sizeof(SceneInfo) << std::endl;
	std::cout <<"GPU: Ray               : " << sizeof(Ray) << std::endl;
	std::cout <<"GPU: PrimitiveType     : " << sizeof(PrimitiveType) << std::endl;
	std::cout <<"GPU: Material          : " << sizeof(Material) << std::endl;
	std::cout <<"GPU: BoundingBox       : " << sizeof(BoundingBox) << std::endl;
	std::cout <<"GPU: Primitive         : " << sizeof(Primitive) << std::endl;
	std::cout <<"GPU: PostProcessingType: " << sizeof(PostProcessingType) << std::endl;
	std::cout <<"GPU: PostProcessingInfo: " << sizeof(PostProcessingInfo) << std::endl;
	std::cout <<"Textures " << NB_MAX_TEXTURES << std::endl;
#endif // 0
}

/*
________________________________________________________________________________

GPU finalization
________________________________________________________________________________
*/
extern "C" void finalize_scene()
{
	checkCudaErrors(cudaFree( d_boundingBoxes ));
	checkCudaErrors(cudaFree( d_primitives ));
	checkCudaErrors(cudaFree( d_lamps ));
	checkCudaErrors(cudaFree( d_materials ));
	checkCudaErrors(cudaFree( d_textures ));
	checkCudaErrors(cudaFree( d_randoms ));
	checkCudaErrors(cudaFree( d_postProcessingBuffer ));
	checkCudaErrors(cudaFree( d_bitmap ));
	checkCudaErrors(cudaFree( d_primitivesXYIds ));
}

/*
________________________________________________________________________________

CPU -> GPU data transfers
________________________________________________________________________________
*/
extern "C" void h2d_scene( 
	BoundingBox* boundingBoxes, int nbActiveBoxes,
	Primitive*  primitives, int nbPrimitives,
	int* lamps, int nbLamps )
{
	checkCudaErrors(cudaMemcpy( d_boundingBoxes,      boundingBoxes,      nbActiveBoxes*sizeof(BoundingBox), cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy( d_primitives,         primitives,         nbPrimitives*sizeof(Primitive),    cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy( d_lamps,              lamps,              nbLamps*sizeof(int),               cudaMemcpyHostToDevice ));
}

extern "C" void h2d_materials( 
	Material*  materials, int nbActiveMaterials,
	char*      textures , int nbActiveTextures,
	float*     randoms,   int nbRandoms)
{
	checkCudaErrors(cudaMemcpy( d_materials, materials, nbActiveMaterials*sizeof(Material), cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy( d_textures,  textures,  gTextureOffset+nbActiveTextures*sizeof(char)*gTextureSize,  cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy( d_randoms,   randoms,   nbRandoms*sizeof(float), cudaMemcpyHostToDevice ));
}

#ifdef USE_KINECT
extern "C" void h2d_kinect( 
	char* kinectVideo, char* kinectDepth )
{
	checkCudaErrors(cudaMemcpy( d_textures, kinectVideo, gKinectVideoSize*sizeof(char), cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy( d_textures+gKinectVideoSize, kinectDepth, gKinectDepthSize*sizeof(char), cudaMemcpyHostToDevice ));
}
#endif // USE_KINECT

/*
________________________________________________________________________________

GPU -> CPU data transfers
________________________________________________________________________________
*/
extern "C" void d2h_bitmap( char* bitmap, int* primitivesXYIds, const SceneInfo sceneInfo )
{
	checkCudaErrors(cudaMemcpy( bitmap, d_bitmap, sceneInfo.width.x*sceneInfo.height.x*gColorDepth*sizeof(char), cudaMemcpyDeviceToHost ));
	checkCudaErrors(cudaMemcpy( primitivesXYIds, d_primitivesXYIds, sceneInfo.width.x*sceneInfo.height.x*sizeof(int), cudaMemcpyDeviceToHost ));
}

/*
________________________________________________________________________________

Kernel launcher
________________________________________________________________________________
*/
extern "C" void cudaRender(
	int4 blockSize, int sharedMemSize,
	SceneInfo sceneInfo,
	int4 objects,
	PostProcessingInfo postProcessingInfo,
	float4 origin, 
	float4 direction, 
	float4 angles)
{
	int2 size;
	size.x = static_cast<int>(sceneInfo.width.x);
	size.y = static_cast<int>(sceneInfo.height.x);

	dim3 grid((size.x+blockSize.x-1)/blockSize.x,(size.y+blockSize.y-1)/blockSize.y,1);
	dim3 blocks( blockSize.x,blockSize.y,blockSize.z );
   sharedMemSize = objects.x*sizeof(BoundingBox);

	switch( sceneInfo.supportFor3DVision.x ) 
	{
	case vtAnaglyph:
		{
			k_anaglyphRenderer<<<grid,blocks,sharedMemSize>>>(
				d_boundingBoxes, objects.x, d_primitives, objects.y,  d_lamps, objects.z, d_materials, d_textures, 
				d_randoms, origin, direction, angles, sceneInfo, 
				postProcessingInfo, d_postProcessingBuffer, d_primitivesXYIds);
			break;
		}
	case vt3DVision:
		{
			k_3DVisionRenderer<<<grid,blocks,sharedMemSize>>>(
				d_boundingBoxes, objects.x, d_primitives, objects.y,  d_lamps, objects.z, d_materials, d_textures, 
				d_randoms, origin, direction, angles, sceneInfo, 
				postProcessingInfo, d_postProcessingBuffer, d_primitivesXYIds);
			break;
		}
	default:
		{
			k_standardRenderer<<<grid,blocks,sharedMemSize>>>(
				d_boundingBoxes, objects.x, d_primitives, objects.y,  d_lamps, objects.z, d_materials, d_textures, 
				d_randoms, origin, direction, angles, sceneInfo,
				postProcessingInfo, d_postProcessingBuffer, d_primitivesXYIds);
			break;
		}
	}

	cudaThreadSynchronize();
	cudaError_t status = cudaGetLastError();
	if(status != cudaSuccess) 
	{
		std::cout << "ERROR: (" << status << ") " << cudaGetErrorString(status) << std::endl;
		std::cout << "INFO: Size(" << size.x << ", " << size.y << ") " << std::endl;
		std::cout << "INFO: Grid(" << grid.x << ", " << grid.y << ", " << grid.z <<") " << std::endl;
		std::cout << "nbActiveBoxes :" << objects.x << std::endl;
		std::cout << "nbActivePrimitives :" << objects.y << std::endl;
		std::cout << "nbActiveLamps :" << objects.z << std::endl;
	}

	switch( postProcessingInfo.type.x )
	{
	case ppe_depthOfField:
		k_depthOfField<<<grid,blocks>>>(
			sceneInfo, 
			postProcessingInfo, 
			d_postProcessingBuffer,
			d_randoms, 
			d_bitmap );
		break;
	case ppe_ambientOcclusion:
		k_ambiantOcclusion<<<grid,blocks>>>(
			sceneInfo, 
			postProcessingInfo, 
			d_postProcessingBuffer,
			d_randoms, 
			d_bitmap );
		break;
	case ppe_cartoon:
		k_cartoon<<<grid,blocks>>>(
			sceneInfo, 
			postProcessingInfo, 
			d_postProcessingBuffer,
			d_randoms, 
			d_bitmap );
		break;
	case ppe_antiAliasing:
		k_antiAliasing<<<grid,blocks>>>(
			sceneInfo, 
			postProcessingInfo, 
			d_postProcessingBuffer,
			d_randoms, 
			d_bitmap );
		break;
	default:
		k_default<<<grid,blocks>>>(
			sceneInfo, 
			postProcessingInfo, 
			d_postProcessingBuffer,
			d_bitmap );
		break;
	}
}
