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
#include <cutil_inline.h>
#include <cutil_math.h>

// Project
#include "CudaDataTypes.h"
#include "../Consts.h"

// Globals
#define gNbIterations 20
#define M_PI 3.14159265358979323846

// Device arrays
Primitive*   d_primitives;
int*         d_boxPrimitivesIndex;
BoundingBox* d_boundingBoxes; 
int*         d_lamps;
Material*    d_materials;
char*        d_textures;
float*       d_randoms;
float4*      d_postProcessingBuffer;
char*        d_bitmap;

#ifdef USE_KINECT
__device__ __constant__ char*        d_kinectVideo;
__device__ __constant__ char*        d_kinectDepth;
#endif // USE_KINECT

// ________________________________________________________________________________
__device__ inline float vectorLength( float4 vector )
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
__device__ inline float dotProduct( const float4 v1, const float4 v2 )
{
   return ( v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
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
__device__ inline void vectorReflection( float4& r, float4& i, float4& n )
{
   r = i-2.f*dotProduct(i,n)*n;
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
__device__ void makeOpenGLColor( 
   const float4 color,
   char*        bitmap,
   int          index)
{
   int mdc_index = index*gColorDepth; 
   bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
   bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
   bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
   bitmap[mdc_index+3] = (char)(color.w*255.f); // Alpha
}

/*
________________________________________________________________________________

Sphere texture Mapping
________________________________________________________________________________
*/
__device__ float4 sphereUVMapping( 
   Primitive& primitive,
   Material*  materials,
   char*      textures,
   float4     intersection,
   float      timer)
{
   float4 result = materials[primitive.materialId.x].color;

   float4 d = primitive.p0-intersection;
   normalizeVector(d);
   int u = primitive.size.x / primitive.materialInfo.x * (0.5f - atan2f(d.z, d.x) / 2*M_PI);
   int v = primitive.size.y / primitive.materialInfo.y * (0.5f - 2.f*(asinf(d.y) / 2*M_PI));

   u = (gTextureOffset+u) % gTextureWidth;
   v = (gTextureOffset+v) % gTextureHeight;
   if( u>=0 && u<gTextureWidth && v>=0 && v<gTextureHeight )
   {
      int index = (materials[primitive.materialId.x].textureId.x*gTextureWidth*gTextureHeight + v*gTextureWidth+u)*gTextureDepth;
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
   Primitive& primitive, 
   Material*  materials,
   char*      textures,
   float4     intersection)
{
   float4 result = materials[primitive.materialId.x].color;
   int x = ((primitive.type.x == ptCheckboard) || (primitive.type.x == ptXZPlane) || (primitive.type.x == ptXYPlane))  ? 
      gTextureOffset+(intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x :
   gTextureOffset+(intersection.z-primitive.p0.z+primitive.size.x)*primitive.materialInfo.x;

   int y = ((primitive.type.x == ptCheckboard) || (primitive.type.x == ptXZPlane)) ? 
      gTextureOffset+(intersection.z+primitive.p0.z+primitive.size.z)*primitive.materialInfo.y :
   gTextureOffset+(intersection.y-primitive.p0.y+primitive.size.z)*primitive.materialInfo.y;

   x = x%gTextureWidth;
   y = y%gTextureHeight;

   if( x>=0 && x<gTextureWidth && y>=0 && y<gTextureHeight )
   {
      int index = (materials[primitive.materialId.x].textureId.x*gTextureWidth*gTextureHeight + y*gTextureWidth+x)*gTextureDepth;
      unsigned char r = textures[index];
      unsigned char g = textures[index+1];
      unsigned char b = textures[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result;
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
   BoundingBox& box, 
   Ray&         ray,
   float        t0,
   float        t1)
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

Sphere intersection
________________________________________________________________________________
*/
__device__ inline bool sphereIntersection(
   SceneInfo& sceneInfo,
   Primitive& sphere, 
   Material*  materials, 
   char*      textures, 
   Ray&       ray, 
   float      timer,
   float4&    intersection,
   float4&    normal,
   float&     shadowIntensity,
   bool&      back
   ) 
{
   // solve the equation sphere-ray to find the intersections
   float4 O_C = ray.origin - sphere.p0;
   normalizeVector(ray.direction);

   float a = 2.f*dotProduct(ray.direction,ray.direction);
   float b = 2.f*dotProduct(O_C,ray.direction);
   float c = dotProduct(O_C,O_C) - (sphere.size.x*sphere.size.x);
   float d = b*b-2.f*a*c;
   if( d>0.f && a != 0.f) 
   {
      float epsilon = 0.9f;

      float r = sqrt(d);
      float t1 = (-b-r)/a;
      float t2 = (-b+r)/a;

      if( t1<=epsilon && t2<=epsilon ) return false; // both intersections are behind the ray origin

      back = (t1<=epsilon || t2<=epsilon); // If only one intersection (t>0) then we are inside the sphere and the intersection is at the back of the sphere

      float t=0.f;
      if( t1<=epsilon ) 
         t = t2;
      else 
         if( t2<=epsilon )
            t = t1;
         else
            t=(t1<t2) ? t1 : t2;
      
      if( t<=epsilon ) return false; // Too close to intersection

      intersection = ray.origin+t*ray.direction; 
      
      // Compute normal vector
      normal = intersection-sphere.p0;
      normal.w = 0.f;
      normal *= (back) ? -1.f : 1.f;
      normalizeVector(normal);

      // Shadow intensity
      shadowIntensity = sceneInfo.shadowIntensity.x*(1.f-materials[sphere.materialId.x].transparency.x);

#if 0
      if( materials[sphere.materialId.x].textured.x == 1) 
      {
         // Procedural texture
         float4 newCenter;
         newCenter.x = sphere.p0.x + 5.f*cos(intersection.x);
         newCenter.y = sphere.p0.y + 5.f*sin(intersection.y);
         newCenter.z = sphere.p0.z + 5.f*sin(cos(intersection.z));
         normal  = intersection - newCenter;
      }
      // Power textures
      if (materials[sphere.materialId.x].textureId.x != NO_TEXTURE && materials[sphere.materialId.x].transparency.x != 0 ) 
      {
         float4 color = sphereUVMapping(sphere, materials, textures, intersection, timer );
         return ((color.x+color.y+color.z) >= sceneInfo.transparentColor.x ); 
      }
#endif // 0

      return true;
   }

#if 0
   // Soft Shadows
   if( result && computingShadows ) 
   {
      float4 O_R;
      O_R.x = ray.x-origin.x;
      O_R.y = ray.y-origin.y;
      O_R.z = ray.z-origin.z;

      normalizeVector(O_R);
      shadowIntensity = dotProduct(O_R, normal);
      shadowIntensity = (shadowIntensity>1.f) ? 1.f : shadowIntensity;
      shadowIntensity = (shadowIntensity<0.f) ? 0.f : shadowIntensity;
   } 
#endif // 0
   return false;
}

/*
________________________________________________________________________________

Cylinder intersection
________________________________________________________________________________
*/
__device__ bool cylinderIntersection( 
   SceneInfo& sceneInfo,
   Primitive& cylinder,
   Material* materials, 
   char*     textures,
   Ray       ray,
   bool      invert,
   float     timer,
   float4&   intersection,
   float4&   normal,
   float&    shadowIntensity,
   bool&     back) 
{
#if 0
   // Top
   bool test = (ray.direction.y<0.f && ray.origin.y>(cylinder.p0.y+cylinder.size.y));
   test = invert ? !test : test;
   if(test) 
   {
      intersection.y = cylinder.p0.y+cylinder.size.y;
      float y = ray.origin.y-cylinder.p0.y-cylinder.size.y;
      intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
      intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
      intersection.w = 1.f; // 1 for top, -1 for bottom

      float4 v=intersection-cylinder.p0;
      v.y = 0.f;

      if( vectorLength(v)<cylinder.size.x ) 
      {
         // Shadow intensity
         shadowIntensity = sceneInfo.shadowIntensity.x*(1.f-materials[cylinder.materialId.x].transparency.x);
         normal.x = 0.f; normal.y = 1.f; normal.z = 0.f;
         return true;
      }
   }
#endif // 0

   /*
   // Bottom
   if( !result && ray.y>0.f && origin.y<(cylinder.p0.y - cylinder.size.y) ) 
   {
      intersection.y = cylinder.p0.y - cylinder.size.y;
      float y = origin.y - cylinder.p0.y + cylinder.size.y;
      intersection.x = origin.x+y*ray.x/-ray.y;
      intersection.z = origin.z+y*ray.z/-ray.y;
      intersection.w = -1.f; // 1 for top, -1 for bottom

      float4 v=intersection-cylinder.p0;
      v.y = 0.f;
      result = (vectorLength(v)<cylinder.size.x);

      normal.x =  0.f;
      normal.y = -1.f;
      normal.z =  0.f;
   }
   */

   float4 O_C = ray.origin - cylinder.p0;
   O_C.y = 0.f;
   float a = 2.f * ( ray.direction.x*ray.direction.x + ray.direction.z*ray.direction.z );
   float b = 2.f*((ray.origin.x-cylinder.p0.x)*ray.direction.x + (ray.origin.z-cylinder.p0.z)*ray.direction.z);
   float c = O_C.x*O_C.x + O_C.z*O_C.z - cylinder.size.x*cylinder.size.x;
   float d = b*b-2.f*a*c;
   if( d!=0.f )
   {
      float epsilon = 0.9f;

      float r = sqrt(d);
      float t1 = (-b-r)/a;
      float t2 = (-b+r)/a;

      if( t1<=epsilon && t2<=epsilon ) return false; // both intersections are behind the ray origin

      back = (t1<=epsilon || t2<=epsilon); // If only one intersection (t>0) then we are inside the sphere and the intersection is at the back of the sphere

      bool twoIntersections(false);
      float t=0.f;
      float tb=0.f;
      if( t1<=epsilon ) 
         t = t2;
      else 
         if( t2<=epsilon )
            t = t1;
         else
         {
            t = (t1<t2) ? t1 : t2;
            tb = (t1>=t2) ? t1 : t2;
            twoIntersections = true;
         }
      
      if( t<=epsilon ) return false; // Too close to intersection

      intersection = ray.origin+t*ray.direction; 
      if( fabs(intersection.y-cylinder.p0.y) > cylinder.size.y ) 
      {
         if( twoIntersections )
         {
            intersection = ray.origin+tb*ray.direction;
            if( fabs(intersection.y-cylinder.p0.y) > cylinder.size.y ) return false;
         }
         else return false;
      }
      
      // Compute normal vector
      normal = intersection-cylinder.p0;
      normal.y = 0.f;
      normal.w = 0.f;
      normal *= (back) ? -1.f : 1.f;
      normalizeVector(normal);

      // Shadow intensity
      shadowIntensity = sceneInfo.shadowIntensity.x*(1.f-materials[cylinder.materialId.x].transparency.x);
      return true;
   }

#if 0
      // Cylinder
      if ( /*d >= 0.f &&*/ a != 0.f) 
      {
         float r = sqrt(d);
         float t1 = (-b-r)/a;
         float t2 = (-b+r)/a;
         float ta = (t1<t2) ? t1 : t2;
         float tb = (t2<t1) ? t1 : t2;

         float4 intersection1;
         float4 intersection2;
         bool i1(false);
         bool i2(false);

         if( ta > 0.f ) 
         {
            // First intersection
            intersection1 = ray.origin+ta*ray.direction;
            intersection1.w = 0.f;
            i1 = ( fabs(intersection1.y - cylinder.p0.y) <= cylinder.size.x );
            // Transparency
            if(i1 && materials[cylinder.materialId.x].textureId.x != NO_TEXTURE ) 
            {
               float4 color = sphereUVMapping(cylinder, materials, textures, intersection1, timer );
               i1 = ((color.x+color.y+color.z) >= sceneInfo.transparentColor.x ); 
            }
         }

         if( tb > 0.f ) 
         {
            // Second intersection
            intersection2 = ray.origin+tb*ray.direction;
            intersection2.w = 0.f;
            i2 = ( fabs(intersection2.y - cylinder.p0.y) <= cylinder.size.x );
            if(i2 && materials[cylinder.materialId.x].textureId.x != NO_TEXTURE ) 
            {
               float4 color = sphereUVMapping(cylinder, materials, textures, intersection2, timer );
               i2 = ((color.x+color.y+color.z) >= sceneInfo.transparentColor.x ); 
            }
         }

         result = i1 || i2;
         if( i1 && i2 )
         {
            float4 O_I1 = intersection1 - ray.origin;
            float4 O_I2 = intersection2 - ray.origin;
            float l1 = vectorLength(O_I1);
            float l2 = vectorLength(O_I2);
            if( l1 < 0.1f ) 
            {
               intersection = intersection2;
            }
            else
            {
               if( l2 < 0.1f )
               {
                  intersection = intersection1;
               }
               else
               {
                  intersection = ( l1<l2 ) ? intersection1 : intersection2;
               }
            }
         }
         else 
         {
            intersection = i1 ? intersection1 : intersection2;
         }
      }
   }

   // Normal to surface
   if( result ) 
   {
      normal   = intersection-cylinder.p0;
      normal.y = 0.f;
      normal.w = 0.f;
      shadowIntensity = 1.f-materials[cylinder.materialId.x].transparency.x;
      if( materials[cylinder.materialId.x].textured.x == 1 ) 
      {
         float4 newCenter;
         newCenter.x = cylinder.p0.x + 5.f*cos(timer*0.58f+intersection.x);
         newCenter.y = cylinder.p0.y + 5.f*sin(timer*0.85f+intersection.y) + intersection.y;
         newCenter.z = cylinder.p0.z + 5.f*sin(cos(timer*1.24f+intersection.z));
         normal = intersection-newCenter;
      }
      normalizeVector( normal );
      result = true;
   }

#endif // 0

   /*
   // Soft Shadows
   if( result && computingShadows ) 
   {
      float4 normal = normalToSurface( cylinder, intersection, depth, materials, timer ); // Normal is computed twice!!!
      normalizeVector(ray );
      normalizeVectornormal;
      shadowIntensity = 5.f*fabs(dotProduct(-ray ,normal));
      shadowIntensity = (shadowIntensity>1.f) ? 1.f : shadowIntensity;
   } 
   */
   return false;
}

/*
________________________________________________________________________________

Checkboard intersection
________________________________________________________________________________
*/
__device__ bool planeIntersection( 
   Primitive& primitive,
   Material* materials,
   char*     textures,
   Ray       ray, 
   bool      reverse,
   float&    shadowIntensity,
   float4&   intersection,
   float4&   normal,
   float     transparentColor,
   float     timer)
{ 
   bool collision = false;

   float reverted = reverse ? -1.f : 1.f;
   switch( primitive.type.x ) 
   {
   case ptMagicCarpet:
   case ptCheckboard:
      {
         intersection.y = primitive.p0.y;
         float y = ray.origin.y-primitive.p0.y;
         if( reverted*ray.direction.y<0.f && reverted*ray.origin.y>reverted*primitive.p0.y) 
         {
            normal.x =  0.f;
            normal.y =  1.f;
            normal.z =  0.f;
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
            normal.x =  0.f;
            normal.y =  1.f;
            normal.z =  0.f;
            intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
            intersection.y = primitive.p0.y;
            intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.z;
         }
         if( !collision && reverted*ray.direction.y>0.f && reverted*ray.origin.y<reverted*primitive.p0.y) 
         {
            normal.x =  0.f;
            normal.y = -1.f;
            normal.z =  0.f;
            intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
            intersection.y = primitive.p0.y;
            intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.z;
         }
         break;
      }
   case ptYZPlane:
      {
         float x = ray.origin.x-primitive.p0.x;
         if( reverted*ray.direction.x<0.f && reverted*ray.origin.x>reverted*primitive.p0.x ) 
         {
            normal.x =  1.f;
            normal.y =  0.f;
            normal.z =  0.f;
            intersection.x = primitive.p0.x;
            intersection.y = ray.origin.y+x*ray.direction.y/-ray.direction.x;
            intersection.z = ray.origin.z+x*ray.direction.z/-ray.direction.x;
            collision = 
               fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.z;
         }
         if( !collision && reverted*ray.direction.x>0.f && reverted*ray.origin.x<reverted*primitive.p0.x ) 
         {
            normal.x = -1.f;
            normal.y =  0.f;
            normal.z =  0.f;
            intersection.x = primitive.p0.x;
            intersection.y = ray.origin.y+x*ray.direction.y/-ray.direction.x;
            intersection.z = ray.origin.z+x*ray.direction.z/-ray.direction.x;
            collision = 
               fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.z;
         }
         break;
      }
   case ptXYPlane:
      {
         float z = ray.origin.z-primitive.p0.z;
         if( reverted*ray.direction.z<0.f && reverted*ray.origin.z>reverted*primitive.p0.z) 
         {
            normal.x =  0.f;
            normal.y =  0.f;
            normal.z =  1.f;
            intersection.z = primitive.p0.z;
            intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
            intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.y - primitive.p0.y) < primitive.size.y;
         }
         if( !collision && reverted*ray.direction.z>0.f && reverted*ray.origin.z<reverted*primitive.p0.z )
         {
            normal.x =  0.f;
            normal.y =  0.f;
            normal.z = -1.f;
            intersection.z = primitive.p0.z;
            intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
            intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.y - primitive.p0.y) < primitive.size.y;
         }
         break;
      }
   case ptCamera:
      {
         if( reverted*ray.direction.z<0.f && reverted*ray.origin.z>reverted*primitive.p0.z )
         {
            normal.x =  0.f;
            normal.y =  0.f;
            normal.z =  1.f;
            intersection.z = primitive.p0.z;
            float z = ray.origin.z-primitive.p0.z;
            intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
            intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
            collision =
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.y - primitive.p0.y) < primitive.size.y;
         }
         break;
      }
   }

   if( collision ) 
   {
      shadowIntensity = 1.f;
      float4 color;
      color = materials[primitive.materialId.x].color;
      if(materials[primitive.materialId.x].textureId.x != NO_TEXTURE)
      {
         color = cubeMapping(primitive, materials, textures, intersection );
      }

      if( materials[primitive.materialId.x].transparency.x != 0.f && ((color.x+color.y+color.z)/3.f) >= transparentColor ) 
      {
         collision = false;
      }
      else 
      {
         shadowIntensity = ((color.x+color.y+color.z)/3.f*(1.f-materials[primitive.materialId.x].transparency.x));
      }
   }
   return collision;
}

#if 0
/*
________________________________________________________________________________

Triangle intersection
________________________________________________________________________________
*/
__device__ bool triangleIntersection( 
   Primitive& triangle, 
   Ray        ray,
   float      timer,
   float4&    intersection,
   float4&    normal,
   bool       computingShadows,
   float&     shadowIntensity,
   float      transparentColor
   ) 
{
   bool result = false;

   float lD = -triangle.p0.x*(triangle.p1.y*triangle.p2.z - triangle.p2.y*triangle.p1.z)
      -triangle.p1.x*(triangle.p2.y*triangle.p0.z - triangle.p0.y*triangle.p2.z)
      -triangle.p2.x*(triangle.p0.y*triangle.p1.z - triangle.p1.y*triangle.p0.z);

   float d = triangle.normal.x*ray.direction.x + triangle.normal.y*ray.direction.y + triangle.normal.z*ray.direction.z;

   d += (d==0.f) ? 0.01f : 0.f;

   float t = -(triangle.normal.x*ray.origin.x + triangle.normal.y*ray.origin.y + triangle.normal.z*ray.origin.z + lD) / d;

   if(t > 0.f)// Triangle in front of the ray
   {
      float4 i = ray.origin+t*ray.direction;

      // 1st side
      float4 I = i - triangle.p0;
      if (dotProduct(triangle.v0,I) <= 0.f)
      {
         // 1st side OK
         I = i - triangle.p1;
         if (dotProduct(triangle.v1,I) <= 0.f)
         {
            // 2nd side OK
            I = i - triangle.p2;
            if (dotProduct(triangle.v2,I) <= 0.f)
            {
               // 1st side OK
               intersection = i;
               normal = triangle.normal;
               result = true;
            }
         }
      }
   }
   return result;
}
#endif // 0

/*
________________________________________________________________________________

Intersection Shader
________________________________________________________________________________
*/
__device__ float4 intersectionShader( 
   SceneInfo& sceneInfo,
   Primitive& primitive, 
   Material*  materials,
   char*      textures,
#ifdef USE_KINECT
   char*      kinectVideo,
#endif // USE_KINECT
   float4     intersection,
   float      timer, 
   bool       back )
{
   float4 colorAtIntersection = materials[primitive.materialId.x].color;
   switch( primitive.type.x ) 
   {
   case ptEnvironment:
   case ptSphere:
      if(materials[primitive.materialId.x].textureId.x != NO_TEXTURE)
      {
         colorAtIntersection = sphereUVMapping(primitive, materials, textures, intersection, timer );
      }
      break;
   case ptCheckboard :
      {
         if( materials[primitive.materialId.x].textureId.x != NO_TEXTURE ) 
         {
            colorAtIntersection = cubeMapping( primitive, materials, textures, intersection );
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
   case ptCylinder:
      {
         if(materials[primitive.materialId.x].textureId.x != NO_TEXTURE)
         {
            colorAtIntersection = sphereUVMapping(primitive, materials, textures, intersection, timer );
         }
         break;
      }
   case ptXYPlane:
   case ptYZPlane:
   case ptXZPlane:
      {
         if( materials[primitive.materialId.x].textureId.x != NO_TEXTURE ) 
         {
            colorAtIntersection = cubeMapping( primitive, materials, textures, intersection );
         }
         break;
      }
#if 0
   case ptTriangle:
      break;
   case ptMagicCarpet:
      {
         if( materials[primitive.materialId.x].textureId.x != NO_TEXTURE ) 
         {
            colorAtIntersection = magicCarpetMapping( primitive, materials, textures, intersection, levels, timer );
         }
         break;
      }
#endif // 0
#ifdef USE_KINECT
   case ptCamera:
      {
         int x = (intersection.x-primitive.p0.x+primitive.size.x)*primitive.materialInfo.x;
         int y = gKinectVideoHeight - (intersection.y-primitive.p0.y+primitive.size.y)*primitive.materialInfo.y;

         x = (x+gKinectVideoWidth)%gKinectVideoWidth;
         y = (y+gKinectVideoHeight)%gKinectVideoHeight;

         if( x>=0 && x<gKinectVideoWidth && y>=0 && y<gKinectVideoHeight ) 
         {
            int index = (y*gKinectVideoWidth+x)*gKinectVideo;
            unsigned char r = kinectVideo[index+2];
            unsigned char g = kinectVideo[index+1];
            unsigned char b = kinectVideo[index+0];
            colorAtIntersection.x = r/256.f;
            colorAtIntersection.y = g/256.f;
            colorAtIntersection.z = b/256.f;
         }
         break;
      }
#endif // USE_KINECT
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
   SceneInfo& sceneInfo,
   BoundingBox* boudingBoxes, int nbActiveBoxes,
   int*       boxPrimitivesIndex,
   Primitive* primitives,
   Material*  materials,
   char*      textures,
   int        nbPrimitives, 
   float4     lampCenter, 
   float4     origin, 
   int        objectId,
   int        iteration,
   float      timer)
{
   float result = 0.f;
   int cptBoxes = 0;
   while( result<=sceneInfo.shadowIntensity.x && cptBoxes < nbActiveBoxes )
   {
      Ray ray;
      ray.origin    = origin;
      ray.direction = lampCenter-origin;
      computeRayAttributes( ray );

      if(boxIntersection(boudingBoxes[cptBoxes], ray, 0.f, sceneInfo.viewDistance.x))
      {
         BoundingBox& box = boudingBoxes[cptBoxes];
         int cptPrimitives = 0;
         while( result<sceneInfo.shadowIntensity.x && cptPrimitives<box.nbPrimitives.x)
         {
            float4 intersection = {0.f,0.f,0.f,0.f};
            float4 normal       = {0.f,0.f,0.f,0.f};
            float  shadowIntensity = 0.f;

            if( boxPrimitivesIndex[box.startIndex.x+cptPrimitives] != objectId && dotProduct(lampCenter,ray.direction) > 0.f )
            {
               Primitive& primitive = primitives[boxPrimitivesIndex[box.startIndex.x+cptPrimitives]];

               bool hit = false;
               bool back;
               switch(primitive.type.x)
               {
               case ptEnvironment :
                  break;
               case ptSphere      : 
                  hit = sphereIntersection  ( sceneInfo, primitive, materials, textures, ray, timer, intersection, normal, shadowIntensity, back ); 
                  break;
               case ptCylinder: 
                  hit = cylinderIntersection( sceneInfo, primitive, materials, textures, ray, true, timer, intersection, normal, shadowIntensity, back ); 
                  break;
   #if 0
               case ptTriangle: 
                  hit = triangleIntersection( primitive, ray, timer, intersection, normal, true, shadowIntensity, sceneInfo.transparentColor ); 
                  break;
   #endif // 0
               default:
                  //TODO: hit = planeIntersection( primitive, materials, textures, ray, true, shadowIntensity, intersection, normal, sceneInfo.transparentColor.x, timer ); 
                  break;
               }
               result += hit ? (shadowIntensity-materials[primitive.materialId.x].innerIllumination.x) : 0.f;
            }
            cptPrimitives++;
         }
      }
      cptBoxes++;
   }
   return (result>1.f) ? 1.f : result;
}

/*
________________________________________________________________________________

Primitive shader
________________________________________________________________________________
*/
__device__ float4 primitiveShader(
   SceneInfo&   sceneInfo,
   BoundingBox* boundingBoxes, int nbActiveBoxes,
   int* boxPrimitivesIndex, Primitive* primitives, int nbActivePrimitives,
   int* lamps, int nbActiveLamps,
   Material* materials, char* textures,
#ifdef USE_KINECT
   char*      kinectVideo,
#endif // USE_KINECT
   float* randoms,
   const float4 origin,
   const float4 normal, 
   const int    objectId, 
   const float4 intersection, 
   const int    iteration,
   const float  timer,
   float4&      refractionFromColor,
   float&       shadowIntensity,
   float4&      totalBlinn)
{
   Primitive primitive = primitives[objectId];
   float4 color = materials[primitive.materialId.x].color;
   float4 lampsColor = { 0.f, 0.f, 0.f, 0.f };

   // Lamp Impact
   float lambert      = 0.f;
   float totalLambert = sceneInfo.backgroundColor.w; // Ambient light
   shadowIntensity    = 0.f;

   //TODO? Lamps have constant color?? if( materials[primitive.materialId.x].innerIllumination.x != 0.f ) return color;

   if( primitive.type.x == ptEnvironment )
   {
      // Final color
      color = intersectionShader( 
         sceneInfo, primitive, materials, textures, 
#ifdef USE_KINECT
         kinectVideo, 
#endif // USE_KINECT
         intersection, timer, false );
   }
   else 
   {
      color *= materials[primitive.materialId.x].innerIllumination.x;
      for( int cptLamps=0; cptLamps<nbActiveLamps; cptLamps++ ) 
      {
         if(lamps[cptLamps] != objectId)
         {
            float4 center = primitives[lamps[cptLamps]].p0;
            if( sceneInfo.pathTracingIteration.x > 0 )
            {
               int t = 3*sceneInfo.pathTracingIteration.x + int(10.f*timer)%100;
               // randomize lamp center
               center.x += primitives[lamps[cptLamps]].size.x*randoms[t  ]*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
               center.y += primitives[lamps[cptLamps]].size.y*randoms[t+1]*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
               center.z += primitives[lamps[cptLamps]].size.z*randoms[t+2]*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
            }

            if( sceneInfo.shadowsEnabled.x ) 
            {
               shadowIntensity = processShadows(
                  sceneInfo, boundingBoxes, nbActiveBoxes,
                  boxPrimitivesIndex, primitives, materials, textures, 
                  nbActivePrimitives, center, 
                  intersection, lamps[cptLamps], iteration, timer );
            }


            float4 lightRay = center - intersection;
            normalizeVector(lightRay);
         
            // Lighted object, not in the shades
            Material& material = materials[primitives[lamps[cptLamps]].materialId.x];
            lampsColor += material.color*material.innerIllumination.x;

            // --------------------------------------------------------------------------------
            // Lambert
            // --------------------------------------------------------------------------------
            lambert = dotProduct(lightRay, normal);
            lambert = (lambert<0.f) ? 0.f : lambert;
            lambert *= (materials[primitive.materialId.x].refraction.x == 0.f) ? material.innerIllumination.x : 1.f;
            lambert *= (1.f-shadowIntensity);
            totalLambert += lambert;

            if( shadowIntensity < sceneInfo.shadowIntensity.x )
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

                  blinnTerm = materials[primitive.materialId.x].specular.x * pow(blinnTerm,materials[primitive.materialId.x].specular.y);
                  totalBlinn += material.color * material.innerIllumination.x * blinnTerm;
               }
            }
         }
      }
      // Final color
      float4 intersectionColor = 
         intersectionShader( sceneInfo, primitive, materials, textures,
#ifdef USE_KINECT
         kinectVideo, 
#endif // USE_KINECT
         intersection, timer, false );

      color += totalLambert*intersectionColor*lampsColor;
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
   SceneInfo& sceneInfo,
   BoundingBox* boundingBoxes, int nbActiveBoxes,
   int* boxPrimitivesIndex, Primitive* primitives, int nbActivePrimitives,
   Material* materials, char* textures,
   Ray     ray, 
   int     iteration,
   float   timer, 
   int&    closestPrimitive, 
   float4& closestIntersection,
   float4& closestNormal,
   bool&   back)
{
   bool intersections = false; 
   float minDistance  = sceneInfo.viewDistance.x;
   Ray r;
   r.origin    = ray.origin;
   r.direction = ray.direction - ray.origin;
   computeRayAttributes( r );

   float4 intersection = {0.f,0.f,0.f,0.f};
   float4 normal       = {0.f,0.f,0.f,0.f};

   for( int cptBoxes = 0; cptBoxes < nbActiveBoxes; ++cptBoxes )
   {
      BoundingBox& box = boundingBoxes[cptBoxes];
      if( boxIntersection(box, r, 0.f, sceneInfo.viewDistance.x/iteration) )
      {
         // Intersection with Box
         if( sceneInfo.renderBoxes.x ) 
         {
            closestPrimitive = cptBoxes;
            return true;
         }
         int cptObjects = 0;
         while( cptObjects<box.nbPrimitives.x)
         { 
            bool i = false;
            float shadowIntensity = 0.f;
            Primitive& primitive = primitives[boxPrimitivesIndex[box.startIndex.x+cptObjects]];

            //float distance = vectorLength( ray.origin - primitive.p0 ) - primitive.size.x; // TODO! Not sure if i should keep it
            //if( distance < minDistance )
            {
               switch( primitive.type.x )
               {
               case ptEnvironment :
               case ptSphere      : 
                  i = sphereIntersection  ( sceneInfo, primitive, materials, textures, r, timer, intersection, normal, shadowIntensity, back ); 
                  break;
               case ptCylinder: 
                  i = cylinderIntersection( sceneInfo, primitive, materials, textures, r, false, timer, intersection, normal, shadowIntensity, back ); 
                  break;
#if 0
               case ptTriangle: 
                  i = triangleIntersection( primitive, r, timer, intersection, normal, false, shadowIntensity, transparentColor ); 
                  break;
#endif // 0
               default        : 
                  i = planeIntersection   ( primitive, materials, textures, r, false, shadowIntensity, intersection, normal, sceneInfo.transparentColor.x, timer); 
                  break;
               }

               if( i ) 
               {
                  float distance = vectorLength( ray.origin - intersection );
                  if(distance>1.f && distance<minDistance) 
                  {
                     // Only keep intersection with the closest object
                     minDistance         = distance;
                     closestPrimitive    = boxPrimitivesIndex[box.startIndex.x+cptObjects];
                     closestIntersection = intersection;
                     closestNormal       = normal;
                     intersections       = true;
                  } 
               }
            }
            cptObjects++;
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
   BoundingBox* boundingBoxes, int nbActiveBoxes,
   int* boxPrimitivesIndex, Primitive* primitives, int nbActivePrimitives,
   int* lamps, int nbActiveLamps,
   Material*  materials, char* textures,
#ifdef USE_KINECT
   char*      kinectVideo, 
#endif // USE_KINECT
   float*     randoms,
   Ray        ray, 
   float      timer, 
   SceneInfo& sceneInfo,
   PostProcessingInfo& postProcessingInfo,
   float4&    intersection,
   float&     depthOfField)
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
   Ray    O_R = ray;
   float4 O_E;
   float4 recursiveColor[gNbIterations+1];
   float4 recursiveRatio[gNbIterations+1];
   float4 recursiveBlinn[gNbIterations+1];

   memset(recursiveRatio,0,sizeof(float4)*(sceneInfo.nbRayIterations.x+1));
   memset(recursiveBlinn,0,sizeof(float4)*(sceneInfo.nbRayIterations.x+1));

   for( int i(0); i<gNbIterations; ++i )
   {
      recursiveColor[i] = sceneInfo.backgroundColor;
   }

   // Variable declarations
   float  shadowIntensity = 0.f;
   float4 refractionFromColor;
   float4 reflectedTarget;
   bool   back;

   while( iteration<sceneInfo.nbRayIterations.x && carryon ) 
   {
      // If no intersection with lamps detected. Now compute intersection with Primitives
      if( carryon ) 
      {
         carryon = intersectionWithPrimitives(
            sceneInfo,
            boundingBoxes, nbActiveBoxes,
            boxPrimitivesIndex, primitives, nbActivePrimitives,
            materials, textures,
            rayOrigin,
            iteration, timer, 
            closestPrimitive, closestIntersection, 
            normal, back);
      }

      if( carryon ) 
      {
         if( sceneInfo.renderBoxes.x ) 
         {
            recursiveColor[iteration] = materials[closestPrimitive%NB_MAX_MATERIALS].color;
         }
         else 
         {
            if ( iteration==0 )
            {
               firstIntersection = closestIntersection;
            }

            // Get object color
            recursiveColor[iteration] = primitiveShader( 
               sceneInfo,
               boundingBoxes, nbActiveBoxes,
               boxPrimitivesIndex, primitives, nbActivePrimitives, lamps, nbActiveLamps, materials, textures, 
   #ifdef USE_KINECT
               kinectVideo, 
   #endif // USE_KINECT
               randoms,
               rayOrigin.origin, normal, closestPrimitive, closestIntersection, 
               iteration, timer, refractionFromColor, shadowIntensity, recursiveBlinn[iteration] );

#if 0 // Distant Transparency
            float4 O_I = ( closestIntersection - ray.origin );
            if( shadowIntensity != 1.f && vectorLength(O_I)<postProcessingInfo.param1.x ) // No reflection/refraction if in shades
#else
            if( shadowIntensity != 1.f ) // No reflection/refraction if in shades
#endif
            {
               // ----------
               // Refraction
               // ----------
               if( materials[primitives[closestPrimitive].materialId.x].transparency.x != 0.f ) 
               {
#if 0
                  // Replace the normal using the intersection color
                  // r,g,b become x,y,z... What the fuck!!
                  if( materials[primitives[closestPrimitive].materialId.x].textureId.x != NO_TEXTURE) 
                  {
                     refractionFromColor -= 0.5f;
                     normal *= refractionFromColor;
                  }
#endif // 0

                  O_E = rayOrigin.origin - closestIntersection;
                  normalizeVector(O_E);
                  float refraction = back ? 1.f : materials[primitives[closestPrimitive].materialId.x].refraction.x;
                  vectorRefraction( O_R.direction, O_E, refraction, normal, initialRefraction );
                  reflectedTarget = closestIntersection - O_R.direction;

                  recursiveRatio[iteration].x = materials[primitives[closestPrimitive].materialId.x].transparency.x;
                  recursiveRatio[iteration].z = 1.f;

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
                     O_E = rayOrigin.origin - closestIntersection;
                     vectorReflection( O_R.direction, O_E, normal );

                     reflectedTarget = closestIntersection - O_R.direction;

                     recursiveRatio[iteration].x = materials[primitives[closestPrimitive].materialId.x].reflection.x;
                  }
                  else 
                  {
                     carryon = false;
                  }         
               }
            }
            else 
            {
               carryon = false;
            }
            
            rayOrigin.origin    = closestIntersection; 
            rayOrigin.direction = reflectedTarget;

            if( sceneInfo.pathTracingIteration.x != 0 && materials[primitives[closestPrimitive].materialId.x].color.w != 0.f)
            {
               // Randomize view
               int rindex = 3.f*timer + sceneInfo.pathTracingIteration.x;
               rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x);
               rayOrigin.direction.x += randoms[rindex  ]*materials[primitives[closestPrimitive].materialId.x].color.w;
               rayOrigin.direction.y += randoms[rindex+1]*materials[primitives[closestPrimitive].materialId.x].color.w;
               rayOrigin.direction.z += randoms[rindex+2]*materials[primitives[closestPrimitive].materialId.x].color.w;
            }

         }

         iteration++; 
      }
   }

   for( int i=iteration-1; i>=0; --i ) 
   {
      recursiveColor[i] = recursiveColor[i+1]*recursiveRatio[i].x + recursiveColor[i]*(1.f-recursiveRatio[i].x);
      recursiveColor[i] += recursiveBlinn[i];
   }
   intersectionColor = recursiveColor[0];


   saturateVector( intersectionColor );
   intersection = closestIntersection;

   float4 O_I = firstIntersection - ray.origin;
#if EXTENDED_FEATURES
   // --------------------------------------------------
   // Attenation effect (Fog)
   // --------------------------------------------------
   float len = 1.f-(vectorLength(O_I)/sceneInfo.viewDistance.x);
   len = (len>0.f) ? len : 0.f; 
   intersectionColor.x = intersectionColor.x * len;
   intersectionColor.y = intersectionColor.y * len;
   intersectionColor.z = intersectionColor.z * len;
#endif // 0

   // Depth of field
   //float4 FI_I = firstIntersection - ray.direction;
   depthOfField = (vectorLength(O_I)-depthOfField)/sceneInfo.viewDistance.x;
   return intersectionColor;
}


/*
________________________________________________________________________________

Standard renderer
________________________________________________________________________________
*/
__global__ void k_standardRenderer(
   BoundingBox* BoundingBoxes, int nbActiveBoxes,
   int* boxPrimitivesIndex, Primitive* primitives, int nbActivePrimitives,
   int* lamps, int nbActiveLamps,
   Material*    materials,
   char*        textures,
#ifdef USE_KINECT
   char*        kinectVideo,
#endif // USE_KINECT
   float*       randoms,
   Ray          ray,
   float4       angles,
   SceneInfo    sceneInfo,
   float        timer,
   PostProcessingInfo postProcessingInfo,
   float4*      postProcessingBuffer)
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
   else
   {
      // Randomize view
      int rindex = index + 3.f*timer + sceneInfo.pathTracingIteration.x;
      rindex = rindex%(sceneInfo.width.x*sceneInfo.height.x);
      ray.direction.x += randoms[rindex  ]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
      ray.direction.y += randoms[rindex+1]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
      ray.direction.z += randoms[rindex+2]*postProcessingBuffer[index].w*postProcessingInfo.param2.x*sceneInfo.pathTracingIteration.x/float(sceneInfo.maxPathTracingIterations.x);
   }

   float dof = postProcessingInfo.param1.x;
   float4 intersection;
   

#if 0 // Isometric 3D
   ray.direction.x = ray.direction.x - (ray.origin.z*0.01f)*(float)(x - (sceneInfo.width.x/2));
   ray.direction.y = ray.direction.y + (ray.origin.z*0.01f)*(float)(y - (sceneInfo.height.x/2));
   ray.origin.x = ray.direction.x;
   ray.origin.y = ray.direction.y;
#else
   ray.direction.x = ray.direction.x - 8.f*(float)(x - (sceneInfo.width.x/2));
   ray.direction.y = ray.direction.y + 8.f*(float)(y - (sceneInfo.height.x/2));
#endif // 0

   vectorRotation( ray.origin, rotationCenter, angles );
   vectorRotation( ray.direction, rotationCenter, angles );

   float4 color = launchRay(
      BoundingBoxes, nbActiveBoxes,
      boxPrimitivesIndex, primitives, nbActivePrimitives,
      lamps, nbActiveLamps,
      materials, textures, 
#ifdef USE_KINECT
      kinectVideo, 
#endif // USE_KINECT
      randoms,
      ray, timer, 
      sceneInfo,
      postProcessingInfo,
      intersection,
      dof);
   
   postProcessingBuffer[index].x += color.x;
   postProcessingBuffer[index].y += color.y;
   postProcessingBuffer[index].z += color.z;
   if( sceneInfo.pathTracingIteration.x == 0 ) postProcessingBuffer[index].w = dof;
}

/*
________________________________________________________________________________

Anaglyph Renderer
________________________________________________________________________________
*/
__global__ void k_anaglyphRenderer(
   BoundingBox* BoundingBoxes, int nbActiveBoxes,
   int* boxPrimitivesIndex, Primitive* primitives, int nbActivePrimitives,
   int* lamps, int nbActiveLamps,
   Material*    materials,
   char*        textures,
#ifdef USE_KINECT
   char*        kinectVideo,
#endif // USE_KINECT
   float*       randoms,
   Ray          ray,
   float4       angles,
   SceneInfo    sceneInfo,
   float        timer,
   PostProcessingInfo postProcessingInfo,
   float4*      postProcessingBuffer)
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
   eyeRay.origin.x = ray.origin.x + sceneInfo.width3DVision.x;
   eyeRay.origin.y = ray.origin.y;
   eyeRay.origin.z = ray.origin.z;

   eyeRay.direction.x = ray.direction.x - 8.f*(float)(x - (sceneInfo.width.x/2));
   eyeRay.direction.y = ray.direction.y + 8.f*(float)(y - (sceneInfo.height.x/2));
   eyeRay.direction.z = ray.direction.z;

   vectorRotation( eyeRay.origin, rotationCenter, angles );
   vectorRotation( eyeRay.direction, rotationCenter, angles );

   float4 colorLeft = launchRay(
      BoundingBoxes, nbActiveBoxes,
      boxPrimitivesIndex, primitives, nbActivePrimitives,
      lamps, nbActiveLamps,
      materials, textures, 
#ifdef USE_KINECT
      kinectVideo, 
#endif // USE_KINECT
      randoms,
      eyeRay, timer, 
      sceneInfo,
      postProcessingInfo,
      intersection,
      dof);

   // Right eye
   eyeRay.origin.x = ray.origin.x - sceneInfo.width3DVision.x;
   eyeRay.origin.y = ray.origin.y;
   eyeRay.origin.z = ray.origin.z;

   eyeRay.direction.x = ray.direction.x - 8.f*(float)(x - (sceneInfo.width.x/2));
   eyeRay.direction.y = ray.direction.y + 8.f*(float)(y - (sceneInfo.height.x/2));
   eyeRay.direction.z = ray.direction.z;

   vectorRotation( eyeRay.origin, rotationCenter, angles );
   vectorRotation( eyeRay.direction, rotationCenter, angles );
   float4 colorRight = launchRay(
      BoundingBoxes, nbActiveBoxes,
      boxPrimitivesIndex, primitives, nbActivePrimitives,
      lamps, nbActiveLamps,
      materials, textures, 
#ifdef USE_KINECT
      kinectVideo, 
#endif // USE_KINECT
      randoms,
      eyeRay, timer, 
      sceneInfo,
      postProcessingInfo,
      intersection,
      dof);

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
   int* boxPrimitivesIndex, Primitive*   primitives,    int nbActivePrimitives,
   int* lamps, int nbActiveLamps,
   Material*    materials,
   char*        textures,
#ifdef USE_KINECT
   char*        kinectVideo,
#endif // USE_KINECT
   float*       randoms,
   Ray          ray,
   float4       angles,
   SceneInfo    sceneInfo,
   float        timer,
   PostProcessingInfo postProcessingInfo,
   float4*      postProcessingBuffer)
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
      eyeRay.origin.x = ray.origin.x + sceneInfo.width3DVision.x;
      eyeRay.origin.y = ray.origin.y;
      eyeRay.origin.z = ray.origin.z;

      eyeRay.direction.x = ray.direction.x - 8.f*(float)(x - (sceneInfo.width.x/2) + halfWidth/2 );
      eyeRay.direction.y = ray.direction.y + 8.f*(float)(y - (sceneInfo.height.x/2));
      eyeRay.direction.z = ray.direction.z;
   }
   else
   {
      // Right eye
      eyeRay.origin.x = ray.origin.x - sceneInfo.width3DVision.x;
      eyeRay.origin.y = ray.origin.y;
      eyeRay.origin.z = ray.origin.z;

      eyeRay.direction.x = ray.direction.x - 8.f*(float)(x - (sceneInfo.width.x/2) - halfWidth/2);
      eyeRay.direction.y = ray.direction.y + 8.f*(float)(y - (sceneInfo.height.x/2));
      eyeRay.direction.z = ray.direction.z;
   }
      
   vectorRotation( eyeRay.origin, rotationCenter, angles );
   vectorRotation( eyeRay.direction, rotationCenter, angles );

   float4 color = launchRay(
      BoundingBoxes, nbActiveBoxes,
      boxPrimitivesIndex, primitives, nbActivePrimitives,
      lamps, nbActiveLamps,
      materials, textures, 
#ifdef USE_KINECT
      kinectVideo, 
#endif // USE_KINECT
      randoms,
      eyeRay, timer, 
      sceneInfo,
      postProcessingInfo,
      intersection,
      dof);

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
   localColor.w = 0.f;

   makeOpenGLColor( localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Ambiant Occlusion
________________________________________________________________________________
*/
__global__ void k_ambiantOcclusion(
   SceneInfo        sceneInfo,
   PostProcessingInfo PostProcessingInfo,
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

   const int step = 3;
   for( int X=-step; X<step; ++X )
   {
      for( int Y=-step; Y<step; ++Y )
      {
         int xx = x+X;
         int yy = y+Y;
         if( xx>=0 && xx<sceneInfo.width.x && yy>=0 && yy<sceneInfo.height.x )
         {
            int localIndex = yy*sceneInfo.width.x+xx;
            if( postProcessingBuffer[localIndex].w >= depth )
            {
               occ += 1.f;
            }
         }
      }
   }
   occ /= float((2*step)*(2*step));
   occ += 0.5f; // Ambient light
   localColor.x *= occ;
   localColor.y *= occ;
   localColor.z *= occ;
   localColor /= (sceneInfo.pathTracingIteration.x+1);
   saturateVector( localColor );
   localColor.w = 0.f;

   makeOpenGLColor( localColor, bitmap, index ); 
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

   localColor.w = 0.f;
   makeOpenGLColor( localColor, bitmap, index ); 
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
   localColor.w = 0.f;
   makeOpenGLColor( localColor, bitmap, index ); 
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

   float4 color = postProcessingBuffer[index]/(sceneInfo.pathTracingIteration.x+1);
   makeOpenGLColor( color, bitmap, index ); 
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
   cutilSafeCall(cudaMalloc( (void**)&d_boundingBoxes,      NB_MAX_BOXES*sizeof(BoundingBox)));
   cutilSafeCall(cudaMalloc( (void**)&d_boxPrimitivesIndex, nbPrimitives*sizeof(int)));
   cutilSafeCall(cudaMalloc( (void**)&d_primitives,         nbPrimitives*sizeof(Primitive)));
   cutilSafeCall(cudaMalloc( (void**)&d_lamps,              nbLamps*sizeof(int)));
   cutilSafeCall(cudaMalloc( (void**)&d_materials,          nbMaterials*sizeof(Material)));
   cutilSafeCall(cudaMalloc( (void**)&d_textures,           nbTextures*gTextureDepth*gTextureWidth*gTextureHeight));
   cutilSafeCall(cudaMalloc( (void**)&d_randoms,            width*height*sizeof(float)));

   // Rendering canvas
   cutilSafeCall(cudaMalloc( (void**)&d_postProcessingBuffer,  width*height*sizeof(float4)));
   cutilSafeCall(cudaMalloc( (void**)&d_bitmap,                width*height*gColorDepth*sizeof(char)));

#ifdef USE_KINECT
   // Kinect video and depth buffers
   cutilSafeCall(cudaMalloc( (void**)&d_kinectVideo,   gKinectVideo*gKinectVideoWidth*gKinectVideoHeight*sizeof(char)));
   cutilSafeCall(cudaMalloc( (void**)&d_kinectDepth,   gKinectDepth*gKinectDepthWidth*gKinectDepthHeight*sizeof(char)));
#endif // USE_KINECT

   std::cout << "GPU: SceneInfo         : " << sizeof(SceneInfo) << std::endl;
   std::cout << "GPU: Ray               : " << sizeof(Ray) << std::endl;
   std::cout << "GPU: PrimitiveType     : " << sizeof(PrimitiveType) << std::endl;
   std::cout << "GPU: Material          : " << sizeof(Material) << std::endl;
   std::cout << "GPU: BoundingBox       : " << sizeof(BoundingBox) << std::endl;
   std::cout << "GPU: Primitive         : " << sizeof(Primitive) << std::endl;
   std::cout << "GPU: PostProcessingType: " << sizeof(PostProcessingType) << std::endl;
   std::cout << "GPU: PostProcessingInfo: " << sizeof(PostProcessingInfo) << std::endl;

   std::cout << NB_MAX_BOXES << " boxes" << std::endl;
   std::cout << nbPrimitives << " primitives" << std::endl;
   std::cout << nbLamps << " lamps" << std::endl;
   std::cout << nbMaterials << " materials" << std::endl;
   std::cout << nbTextures << " textures" << std::endl;
}

/*
________________________________________________________________________________

GPU finalization
________________________________________________________________________________
*/
extern "C" void finalize_scene()
{
   cutilSafeCall(cudaFree( d_boundingBoxes ));
   cutilSafeCall(cudaFree( d_boxPrimitivesIndex ));
   cutilSafeCall(cudaFree( d_primitives ));
   cutilSafeCall(cudaFree( d_lamps ));
   cutilSafeCall(cudaFree( d_materials ));
   cutilSafeCall(cudaFree( d_textures ));
   cutilSafeCall(cudaFree( d_randoms ));
   cutilSafeCall(cudaFree( d_postProcessingBuffer ));
   cutilSafeCall(cudaFree( d_bitmap ));
#ifdef USE_KINECT
   cutilSafeCall(cudaFree( d_kinectVideo ));
   cutilSafeCall(cudaFree( d_kinectDepth ));
#endif // USE_KINECT
}

/*
________________________________________________________________________________

CPU -> GPU data transfers
________________________________________________________________________________
*/
extern "C" void h2d_scene( 
   BoundingBox* boundingBoxes, int nbActiveBoxes,
   int* boxPrimitivesIndex, Primitive*  primitives, int nbPrimitives,
   int* lamps, int nbLamps )
{
   cutilSafeCall(cudaMemcpy( d_boundingBoxes,      boundingBoxes,      nbActiveBoxes*sizeof(BoundingBox), cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_boxPrimitivesIndex, boxPrimitivesIndex, nbPrimitives*sizeof(int),          cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_primitives,         primitives,         nbPrimitives*sizeof(Primitive),    cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_lamps,              lamps,              nbLamps*sizeof(int),               cudaMemcpyHostToDevice ));
}

extern "C" void h2d_materials( 
   Material*  materials, int nbActiveMaterials,
   char*      textures , int nbActiveTextures,
   float*     randoms,   int nbRandoms)
{
   cutilSafeCall(cudaMemcpy( d_materials, materials, nbActiveMaterials*sizeof(Material), cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_textures,  textures,  nbActiveTextures*sizeof(char)*gTextureDepth*gTextureWidth*gTextureHeight,  cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_randoms,   randoms,   nbRandoms*sizeof(float), cudaMemcpyHostToDevice ));
}

#ifdef USE_KINECT
extern "C" void h2d_kinect( 
   char* kinectVideo, int videoSize,
   char* kinectDepth, int depthSize )
{
   cutilSafeCall(cudaMemcpy( d_kinectVideo, kinectVideo, videoSize*sizeof(char), cudaMemcpyHostToDevice ));
   cutilSafeCall(cudaMemcpy( d_kinectDepth, kinectDepth, depthSize*sizeof(char), cudaMemcpyHostToDevice ));
}
#endif // USE_KINECT

/*
________________________________________________________________________________

GPU -> CPU data transfers
________________________________________________________________________________
*/
extern "C" void d2h_bitmap( char* bitmap, const SceneInfo sceneInfo )
{
   cutilSafeCall(cudaMemcpy( bitmap, d_bitmap, sceneInfo.width.x*sceneInfo.height.x*gColorDepth*sizeof(char), cudaMemcpyDeviceToHost ));
}

/*
________________________________________________________________________________

Kernel launcher
________________________________________________________________________________
*/
extern "C" void cudaRender(
   dim3 blockSize, int sharedMemSize,
   SceneInfo sceneInfo,
   int4 objects,
   PostProcessingInfo postProcessingInfo,
   float timer,
   Ray ray, 
   float4 angles)
{
   int2 size;
   size.x = static_cast<int>(sceneInfo.width.x);
   size.y = static_cast<int>(sceneInfo.height.x);

   dim3 grid((size.x+blockSize.x-1)/blockSize.x,(size.y+blockSize.y-1)/blockSize.y,1);

   switch( sceneInfo.supportFor3DVision.x ) 
   {
   case vtAnaglyph:
      {
         k_anaglyphRenderer<<<grid,blockSize,sharedMemSize>>>(
            d_boundingBoxes, objects.x, d_boxPrimitivesIndex, d_primitives, objects.y,  d_lamps, objects.z, d_materials, d_textures, 
#ifdef USE_KINECT
            d_kinectVideo, 
#endif // USE_KINECT
            d_randoms,ray, angles, sceneInfo, timer, postProcessingInfo, d_postProcessingBuffer);
         break;
      }
   case vt3DVision:
      {
         k_3DVisionRenderer<<<grid,blockSize,sharedMemSize>>>(
            d_boundingBoxes, objects.x, d_boxPrimitivesIndex, d_primitives, objects.y,  d_lamps, objects.z, d_materials, d_textures, 
#ifdef USE_KINECT
            d_kinectVideo, 
#endif // USE_KINECT
            d_randoms,ray, angles, sceneInfo, timer, postProcessingInfo, d_postProcessingBuffer);
         break;
      }
   default:
      {
         k_standardRenderer<<<grid,blockSize,sharedMemSize>>>(
            d_boundingBoxes, objects.x, d_boxPrimitivesIndex, d_primitives, objects.y,  d_lamps, objects.z, d_materials, d_textures, 
#ifdef USE_KINECT
            d_kinectVideo, 
#endif // USE_KINECT
            d_randoms,ray, angles, sceneInfo, timer, postProcessingInfo, d_postProcessingBuffer);
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
   }

   switch( postProcessingInfo.type.x )
   {
   case ppe_depthOfField:
      k_depthOfField<<<grid,blockSize>>>(
         sceneInfo, 
         postProcessingInfo, 
         d_postProcessingBuffer,
         d_randoms, 
         d_bitmap );
      break;
   case ppe_ambientOcclusion:
      k_ambiantOcclusion<<<grid,blockSize>>>(
         sceneInfo, 
         postProcessingInfo, 
         d_postProcessingBuffer,
         d_randoms, 
         d_bitmap );
         break;
   case ppe_cartoon:
      k_cartoon<<<grid,blockSize>>>(
         sceneInfo, 
         postProcessingInfo, 
         d_postProcessingBuffer,
         d_randoms, 
         d_bitmap );
      break;
   case ppe_antiAliasing:
      k_antiAliasing<<<grid,blockSize>>>(
         sceneInfo, 
         postProcessingInfo, 
         d_postProcessingBuffer,
         d_randoms, 
         d_bitmap );
      break;
   default:
      k_default<<<grid,blockSize>>>(
         sceneInfo, 
         postProcessingInfo, 
         d_postProcessingBuffer,
         d_bitmap );
      break;
   }
}
