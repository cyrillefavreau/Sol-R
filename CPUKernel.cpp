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

// System
#include <iostream>

// OpenGL
#include <GL/freeglut.h>

// Project
#include "Cuda/CudaDataTypes.h"
#include "Consts.h"
#include "Logging.h"

#include "CPUKernel.h"

//#define PHOTON_ENERGY
#define GRADIANT_BACKGROUND

CPUKernel::CPUKernel( bool activeLogging, int optimalNbOfPrimmitivesPerBox, int platform, int device ) 
   : GPUKernel( activeLogging, optimalNbOfPrimmitivesPerBox, platform, device ), 
   m_postProcessingBuffer(nullptr)
{
}


CPUKernel::~CPUKernel(void)
{
   if( m_postProcessingBuffer ) delete m_postProcessingBuffer;
}

// ________________________________________________________________________________
void CPUKernel::saturateVector( float4& v )
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
float3 CPUKernel::crossProduct( const float3& b, const float3& c )
{
   float3 a;
   a.x = b.y*c.z - b.z*c.y;
   a.y = b.z*c.x - b.x*c.z;
   a.z = b.x*c.y - b.y*c.x;
   return a;
}

float CPUKernel::vectorLength( const float3& v )
{
   return sqrtf( v.x*v.x+v.y*v.y+v.z*v.z );
}

float CPUKernel::dot( const float3& v1, const float3& v2 )
{
   return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
}

float3 CPUKernel::normalize( const float3& v )
{
   float3 returnValue;
   float len=vectorLength(v);
   returnValue.x = v.x/len;
   returnValue.y = v.y/len;
   returnValue.z = v.z/len;
   return returnValue;
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
void CPUKernel::vectorReflection( float3& r, const float3& i, const float3& n )
{
   r.x = i.x-2.f*dot(i,n)*n.x;
   r.y = i.y-2.f*dot(i,n)*n.y;
   r.z = i.z-2.f*dot(i,n)*n.z;
}

/*
________________________________________________________________________________
incident: le vecteur norm? inverse ? la direction d?incidence de la source 
lumineuse
n1      : index of refraction of original medium
n2      : index of refraction of new medium
________________________________________________________________________________
*/
void CPUKernel::vectorRefraction( 
   float3&      refracted, 
   const float3 incident, 
   const float  n1, 
   const float3 normal, 
   const float  n2 )
{
   refracted = incident;
   if(n1!=n2 && n2!=0.f) 
   {
      float r = n1/n2;
      float cosI = dot( incident, normal );
      float cosT2 = 1.f - r*r*(1.f - cosI*cosI);
      refracted.x = r*incident.x + (r*cosI-sqrt( fabs(cosT2) ))*normal.x;
      refracted.y = r*incident.y + (r*cosI-sqrt( fabs(cosT2) ))*normal.y;
      refracted.z = r*incident.z + (r*cosI-sqrt( fabs(cosT2) ))*normal.z;
   }
}

/*
________________________________________________________________________________
__v : Vector to rotate
__c : Center of rotations
__a : m_angles
________________________________________________________________________________
*/
void CPUKernel::vectorRotation( float3& v, const float3& rotationCenter, const float3& m_angles )
{ 
   float3 cosAngles, sinAngles;

   cosAngles.x = cos(m_angles.x);
   cosAngles.y = cos(m_angles.y);
   cosAngles.z = cos(m_angles.z);

   sinAngles.x = sin(m_angles.x);
   sinAngles.y = sin(m_angles.y);
   sinAngles.z = sin(m_angles.z);

   // Rotate Center
   float3 vector;
   vector.x = v.x - rotationCenter.x;
   vector.y = v.y - rotationCenter.y;
   vector.z = v.z - rotationCenter.z;
   float3 result = vector; 

   /* X axis */ 
   result.y = vector.y*cosAngles.x - vector.z*sinAngles.x; 
   result.z = vector.y*sinAngles.x + vector.z*cosAngles.x; 
   vector = result; 
   result = vector; 

   /* Y axis */ 
   result.z = vector.z*cosAngles.y - vector.x*sinAngles.y; 
   result.x = vector.z*sinAngles.y + vector.x*cosAngles.y; 
   vector = result; 
   result = vector; 

   /* Z axis */ 
   result.x = vector.x*cosAngles.z - vector.y*sinAngles.z; 
   result.y = vector.x*sinAngles.z + vector.y*cosAngles.z; 

   v.x = result.x + rotationCenter.x; 
   v.y = result.y + rotationCenter.y; 
   v.z = result.z + rotationCenter.z;
}

/*
________________________________________________________________________________

Compute ray attributes
________________________________________________________________________________
*/
void CPUKernel::computeRayAttributes(Ray& ray)
{
   ray.inv_direction.x = 1.f/ray.direction.x;
   ray.inv_direction.y = 1.f/ray.direction.y;
   ray.inv_direction.z = 1.f/ray.direction.z;
   ray.signs.x = (ray.inv_direction.x < 0);
   ray.signs.y = (ray.inv_direction.y < 0);
   ray.signs.z = (ray.inv_direction.z < 0);
}

void CPUKernel::juliaSet( 
   const Primitive& primitive,
   const float x, 
   const float y, 
   float4& color )
{
   Material& material=m_hMaterials[primitive.materialId.x];
   float W = (float)material.textureMapping.x;
   float H = (float)material.textureMapping.y;

   //pick some values for the constant c, this determines the shape of the Julia Set
   float cRe = -0.7f + 0.4f*sinf(m_sceneInfo.misc.y/1500.f);
   float cIm = 0.27015f + 0.4f*cosf(m_sceneInfo.misc.y/2000.f);

   //calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
   float newRe = 1.5f * (x - W / 2.f) / (0.5f * W);
   float newIm = (y - H / 2.f) / (0.5f * H);
   //i will represent the number of iterations
   int n;
   //start the iteration process
   float  maxIterations = 40.f+m_sceneInfo.pathTracingIteration.x;
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
   color.w = 1.f-(n/maxIterations);
}

void CPUKernel::mandelbrotSet( 
   const Primitive& primitive,
   const float x, 
   const float y, 
   float4& color )
{
   Material& material=m_hMaterials[primitive.materialId.x];
   float W = (float)material.textureMapping.x;
   float H = (float)material.textureMapping.y;

   float  MinRe		= -2.f;
   float  MaxRe		=	1.f;
   float  MinIm		= -1.2f;
   float  MaxIm		=	MinIm + (MaxRe - MinRe) * H/W;
   float  Re_factor	=	(MaxRe - MinRe) / (W - 1.f);
   float  Im_factor	=	(MaxIm - MinIm) / (H - 1.f);
   float  maxIterations = (float)(NB_MAX_ITERATIONS+m_sceneInfo.pathTracingIteration.x);

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

   color.x = 1.f-color.x*(n/maxIterations);
   color.y = 1.f-color.y*(n/maxIterations);
   color.z = 1.f-color.z*(n/maxIterations);
   color.w = 1.f-(n/maxIterations);
}

/*
________________________________________________________________________________

Sphere texture Mapping
________________________________________________________________________________
*/
float4 CPUKernel::sphereUVMapping( 
   const Primitive& primitive,
   const float3&    intersection)
{
   Material& material=m_hMaterials[primitive.materialId.x];
   float4 result = material.color;

   float3 d;
   d.x = primitive.p0.x-intersection.x;
   d.y = primitive.p0.y-intersection.y;
   d.z = primitive.p0.z-intersection.z;
   d = normalize(d);

   int u = int(primitive.size.x * (0.5f - atan2f(d.z, d.x) / 2.f*PI));
   int v = int(primitive.size.y * (0.5f - 2.f*(asinf(d.y) / 2.f*PI)));

   if( material.textureMapping.x != 0 ) u = u%material.textureMapping.x;
   if( material.textureMapping.y != 0 ) v = v%material.textureMapping.y;
   if( u>=0 && u<material.textureMapping.x && v>=0 && v<material.textureMapping.y )
   {
      int index = material.textureOffset.x + (v*material.textureMapping.x+u)*material.textureMapping.w;
      index = index%(material.textureMapping.x*material.textureMapping.y*material.textureMapping.w);
      BitmapBuffer r = m_hTextures[material.textureMapping.z].buffer[index  ];
      BitmapBuffer g = m_hTextures[material.textureMapping.z].buffer[index+1];
      BitmapBuffer b = m_hTextures[material.textureMapping.z].buffer[index+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }
   return result; 
}

/*
________________________________________________________________________________

Triangle texture Mapping
________________________________________________________________________________
*/
float4 CPUKernel::triangleUVMapping( 
   const Primitive& primitive,
   const float3&    intersection,
   const float3&    areas)
{
   Material& material=m_hMaterials[primitive.materialId.x];
   float4 result = material.color;

   float3 T;
   T.x = (primitive.vt0.x*areas.x+primitive.vt1.x*areas.y+primitive.vt2.x*areas.z)/(areas.x+areas.y+areas.z);
   T.y = (primitive.vt0.y*areas.x+primitive.vt1.y*areas.y+primitive.vt2.y*areas.z)/(areas.x+areas.y+areas.z);
   T.z = (primitive.vt0.z*areas.x+primitive.vt1.z*areas.y+primitive.vt2.z*areas.z)/(areas.x+areas.y+areas.z);

   int u = int(T.x*material.textureMapping.x);
   int v = int(T.y*material.textureMapping.y);

	u = u%material.textureMapping.x;
	v = v%material.textureMapping.y;

   if( u>=0 && u<material.textureMapping.x && v>=0 && v<material.textureMapping.y )
   {
      switch( material.textureMapping.z )
      {
      case TEXTURE_MANDELBROT: mandelbrotSet( primitive, static_cast<float>(u), static_cast<float>(v), result ); break;
      case TEXTURE_JULIA: juliaSet( primitive, static_cast<float>(u), static_cast<float>(v), result ); break;
      default:
         {
            int index = material.textureOffset.x + (v*material.textureMapping.x+u)*material.textureMapping.w;
            index = index%(material.textureMapping.x*material.textureMapping.y*material.textureMapping.w);
            BitmapBuffer r = m_hTextures[material.textureMapping.z].buffer[index  ];
            BitmapBuffer g = m_hTextures[material.textureMapping.z].buffer[index+1];
            BitmapBuffer b = m_hTextures[material.textureMapping.z].buffer[index+2];
            result.x = r/256.f;
            result.y = g/256.f;
            result.z = b/256.f;
         }
      }
   }
   return result; 
}

/*
________________________________________________________________________________

Cube texture mapping
________________________________________________________________________________
*/
float4 CPUKernel::cubeMapping(
   const Primitive& primitive, 
   const float3&    intersection)
{
   Material& material=m_hMaterials[primitive.materialId.x];
   float4 result = material.color;

#ifdef USE_KINECT
   if( primitive.type.x == ptCamera )
   {
      int x = static_cast<int>((intersection.x-primitive.p0.x+primitive.size.x)*material.textureMapping.x);
      int y = static_cast<int>(gKinectVideoHeight - (intersection.y-primitive.p0.y+primitive.size.y)*material.textureMapping.y);

      x = (x+gKinectVideoWidth)%gKinectVideoWidth;
      y = (y+gKinectVideoHeight)%gKinectVideoHeight;

      if( x>=0 && x<gKinectVideoWidth && y>=0 && y<gKinectVideoHeight ) 
      {
         int index = (y*gKinectVideoWidth+x)*gKinectVideo;
         index = index%(material.textureMapping.x*material.textureMapping.y*material.textureMapping.w);
         BitmapBuffer r = m_hTextures[material.textureMapping.z].buffer[index+2];
         BitmapBuffer g = m_hTextures[material.textureMapping.z].buffer[index+1];
         BitmapBuffer b = m_hTextures[material.textureMapping.z].buffer[index  ];
         result.x = r/256.f;
         result.y = g/256.f;
         result.z = b/256.f;
      }
   }
   else
#endif // USE_KINECT
   {
      int u = ((primitive.type.x == ptCheckboard) || (primitive.type.x == ptXZPlane) || (primitive.type.x == ptXYPlane))  ? 
         static_cast<int>(intersection.x-primitive.p0.x+primitive.size.x):
         static_cast<int>(intersection.z-primitive.p0.z+primitive.size.z);

      int v = ((primitive.type.x == ptCheckboard) || (primitive.type.x == ptXZPlane)) ? 
         static_cast<int>(intersection.z+primitive.p0.z+primitive.size.z) :
         static_cast<int>(intersection.y-primitive.p0.y+primitive.size.y);

      u = u%material.textureMapping.x;
      v = v%material.textureMapping.y;

      if( u>=0 && u<material.textureMapping.x && v>=0 && v<material.textureMapping.x )
      {
         switch( material.textureMapping.z )
         {
         case TEXTURE_MANDELBROT: mandelbrotSet( primitive, static_cast<float>(u), static_cast<float>(v), result ); break;
         case TEXTURE_JULIA: juliaSet( primitive, static_cast<float>(u), static_cast<float>(v), result ); break;
         default:
            {
               int index = material.textureOffset.x + (v*material.textureMapping.x+u)*material.textureMapping.w;
               index = index%(material.textureMapping.x*material.textureMapping.y*material.textureMapping.w);
               BitmapBuffer r = m_hTextures[material.textureMapping.z].buffer[index  ];
               BitmapBuffer g = m_hTextures[material.textureMapping.z].buffer[index+1];
               BitmapBuffer b = m_hTextures[material.textureMapping.z].buffer[index+2];
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

bool CPUKernel::wireFrameMapping( float x, float y, int width, const Primitive& primitive )
{
   int X = static_cast<int>(abs(x));
   int Y = static_cast<int>(abs(y));
   int A = 100; // TODO
   int B = 100; // TODO
   return ( X%A<=width ) || ( Y%B<=width );
}

/*
________________________________________________________________________________

Box intersection
________________________________________________________________________________
*/
bool CPUKernel::boxIntersection( 
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
bool CPUKernel::ellipsoidIntersection(
   const Primitive& ellipsoid,
   const Ray& ray, 
   float3& intersection,
   float3& normal,
   float& shadowIntensity,
   bool& back) 
{
   // Shadow intensity
   shadowIntensity = 1.f;

   // solve the equation sphere-ray to find the intersections
   float3 O_C;
   O_C.x = ray.origin.x-ellipsoid.p0.x;
   O_C.y = ray.origin.y-ellipsoid.p0.y;
   O_C.z = ray.origin.z-ellipsoid.p0.z;
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

   if( t1<=EPSILON && t2<=EPSILON ) return false; // both intersections are behind the ray m_viewPos
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
   intersection.x = ray.origin.x + t*dir.x;
   intersection.y = ray.origin.y + t*dir.y;
   intersection.z = ray.origin.z + t*dir.z;

   normal.x = intersection.x-ellipsoid.p0.x;
   normal.y = intersection.y-ellipsoid.p0.y;
   normal.z = intersection.z-ellipsoid.p0.z;
   normal.x = 2.f*normal.x/(ellipsoid.size.x*ellipsoid.size.x);
   normal.y = 2.f*normal.y/(ellipsoid.size.y*ellipsoid.size.y);
   normal.z = 2.f*normal.z/(ellipsoid.size.z*ellipsoid.size.z);

   normal.x *= (back) ? -1.f : 1.f;
   normal.y *= (back) ? -1.f : 1.f;
   normal.z *= (back) ? -1.f : 1.f;
   normal = normalize(normal);
   return true;
}

/*
________________________________________________________________________________

Sphere intersection
________________________________________________________________________________
*/
bool CPUKernel::sphereIntersection(
   const Primitive& sphere, 
   const Ray& ray, 
   float3&    intersection,
   float3&    normal,
   float&     shadowIntensity,
   bool&      back
   ) 
{
   // solve the equation sphere-ray to find the intersections
   float3 O_C;
   O_C.x = ray.origin.x-sphere.p0.x;
   O_C.y = ray.origin.y-sphere.p0.y;
   O_C.z = ray.origin.z-sphere.p0.z;
   float3 dir = normalize(ray.direction); 

   float a = 2.f*dot(dir,dir);
   float b = 2.f*dot(O_C,dir);
   float c = dot(O_C,O_C) - (sphere.size.x*sphere.size.x);
   float d = b*b-2.f*a*c;

   if( d<=0.f || a == 0.f) return false;
   float r = sqrt(d);
   float t1 = (-b-r)/a;
   float t2 = (-b+r)/a;

   if( t1<=EPSILON && t2<=EPSILON ) return false; // both intersections are behind the ray m_viewPos
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
   intersection.x = ray.origin.x+t*dir.x;
   intersection.y = ray.origin.y+t*dir.y;
   intersection.z = ray.origin.z+t*dir.z;

   // TO REMOVE - For Charts only
   //if( intersection.y < sphere.p0.y ) return false;

   if( m_hMaterials[sphere.materialId.x].attributes.y == 0) 
   {
      // Compute normal vector
      normal.x = intersection.x-sphere.p0.x;
      normal.y = intersection.y-sphere.p0.y;
      normal.z = intersection.z-sphere.p0.z;
   }
   else
   {
      // Procedural texture
      float3 newCenter;
      newCenter.x = sphere.p0.x + 0.008f*sphere.size.x*cos(m_sceneInfo.misc.y + intersection.x );
      newCenter.y = sphere.p0.y + 0.008f*sphere.size.y*sin(m_sceneInfo.misc.y + intersection.y );
      newCenter.z = sphere.p0.z + 0.008f*sphere.size.z*sin(cos(m_sceneInfo.misc.y + intersection.z ));
      normal.x = intersection.x - newCenter.x;
      normal.y = intersection.y - newCenter.y;
      normal.z = intersection.z - newCenter.z;
   }
   normal.x *= (back) ? -1.f : 1.f;
   normal.y *= (back) ? -1.f : 1.f;
   normal.z *= (back) ? -1.f : 1.f;
   normal = normalize(normal);

   shadowIntensity = 1.f;
   // Shadow management
   /*
   r = dot(dir,normal);
   shadowIntensity = (m_hMaterials[sphere.materialId.x].transparency.x != 0.f) ? (1.f-fabs(r)) : 1.f;
   */

#if EXTENDED_FEATURES
   // Power textures
   if (m_hMaterials[sphere.materialId.x].textureInfo.y != TEXTURE_NONE && m_hMaterials[sphere.materialId.x].transparency.x != 0 ) 
   {
      float3 color = sphereUVMapping(sphere, m_hMaterials, textures, intersection, timer );
      return ((color.x+color.y+color.z) >= m_sceneInfo.transparentColor.x ); 
   }
#endif // 0

   return true;
}

/*
________________________________________________________________________________

Cylinder intersection
________________________________________________________________________________
*/
bool CPUKernel::cylinderIntersection( 
   const Primitive& cylinder,
   const Ray& ray,
   float3&    intersection,
   float3&    normal,
   float&     shadowIntensity,
   bool&      back) 
{
   back = false;
   float3 center;
   center.x = (cylinder.p0.x+cylinder.p1.x)/2.f;
   center.y = (cylinder.p0.y+cylinder.p1.y)/2.f;
   center.z = (cylinder.p0.z+cylinder.p1.z)/2.f;
   float3 O_C;
   O_C.x = ray.origin.x-center.x;
   O_C.y = ray.origin.y-center.y;
   O_C.z = ray.origin.z-center.z;
   float3 dir = ray.direction;
   float3 n   = crossProduct(dir, cylinder.n1);

   float ln = vectorLength(n);

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
         intersection.x = ray.origin.x+t*dir.x;
         intersection.y = ray.origin.y+t*dir.y;
         intersection.z = ray.origin.z+t*dir.z;

         float3 HB1;
         HB1.x = intersection.x-cylinder.p0.x;
         HB1.y = intersection.y-cylinder.p0.y;
         HB1.z = intersection.z-cylinder.p0.z;
         float3 HB2;
         HB2.x = intersection.x-cylinder.p1.x;
         HB2.y = intersection.y-cylinder.p1.y;
         HB2.z = intersection.z-cylinder.p1.z;

         float scale1 = dot(HB1,cylinder.n1);
         float scale2 = dot(HB2,cylinder.n1);

         // Cylinder length
         if( scale1 < EPSILON || scale2 > EPSILON ) return false;

         if( m_hMaterials[cylinder.materialId.x].attributes.y == 1) 
         {
            // Procedural texture
            float3 newCenter;
            newCenter.x = cylinder.p0.x + 0.01f*cylinder.size.x*cos(m_sceneInfo.misc.y/100.f+intersection.x);
            newCenter.y = cylinder.p0.y + 0.01f*cylinder.size.y*sin(m_sceneInfo.misc.y/100.f+intersection.y);
            newCenter.z = cylinder.p0.z + 0.01f*cylinder.size.z*sin(cos(m_sceneInfo.misc.y/100.f+intersection.z));
            HB1.x = intersection.x - newCenter.x;
            HB1.y = intersection.y - newCenter.y;
            HB1.z = intersection.z - newCenter.z;
         }

         float3 HB;
         HB.x = HB1.x-cylinder.n1.x*scale1;
         HB.y = HB1.y-cylinder.n1.y*scale1;
         HB.z = HB1.z-cylinder.n1.z*scale1;
         normal = normalize(HB);

         shadowIntensity = 1.f;
         /*
         // Shadow management
         dir = normalize(dir);
         float r = dot(dir,normal);
         shadowIntensity = (m_hMaterials[cylinder.materialId.x].transparency.x != 0.f) ? (1.f-fabs(r)) : 1.f;
         */
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
bool CPUKernel::planeIntersection( 
   const Primitive& primitive,
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
         if( reverted*ray.origin.y<0.f && reverted*ray.origin.y>reverted*primitive.p0.y) 
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
         if( reverted*ray.origin.y<0.f && reverted*ray.origin.y>reverted*primitive.p0.y) 
         {
            intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
            intersection.y = primitive.p0.y;
            intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.z;
            if( m_hMaterials[primitive.materialId.x].attributes.z == 2 )  // Wireframe
               collision &= wireFrameMapping(intersection.x, intersection.z, m_hMaterials[primitive.materialId.x].attributes.w, primitive );
         }
         if( !collision && reverted*ray.origin.y>0.f && reverted*ray.origin.y<reverted*primitive.p0.y) 
         {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal.z = -normal.z;
            intersection.x = ray.origin.x+y*ray.direction.x/-ray.direction.y;
            intersection.y = primitive.p0.y;
            intersection.z = ray.origin.z+y*ray.direction.z/-ray.direction.y;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.z;
            if( m_hMaterials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
               collision &= wireFrameMapping(intersection.x, intersection.z, m_hMaterials[primitive.materialId.x].attributes.w, primitive );
         }
         break;
      }
   case ptYZPlane:
      {
         float x = ray.origin.x-primitive.p0.x;
         if( reverted*ray.origin.x<0.f && reverted*ray.origin.x>reverted*primitive.p0.x ) 
         {
            intersection.x = primitive.p0.x;
            intersection.y = ray.origin.y+x*ray.direction.y/-ray.direction.x;
            intersection.z = ray.origin.z+x*ray.direction.z/-ray.direction.x;
            collision = 
               fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.z;
            if( m_hMaterials[primitive.materialId.x].innerIllumination.x != 0.f )
            {
               // Chessboard like Lights
               collision &= int(fabs(intersection.z))%4000<2000 && int(fabs(intersection.y))%4000<2000;
            }
            if( m_hMaterials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
               collision &= wireFrameMapping(intersection.y, intersection.z, m_hMaterials[primitive.materialId.x].attributes.w, primitive );
         }
         if( !collision && reverted*ray.origin.x>0.f && reverted*ray.origin.x<reverted*primitive.p0.x ) 
         {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal.z = -normal.z;
            intersection.x = primitive.p0.x;
            intersection.y = ray.origin.y+x*ray.direction.y/-ray.direction.x;
            intersection.z = ray.origin.z+x*ray.direction.z/-ray.direction.x;
            collision = 
               fabs(intersection.y - primitive.p0.y) < primitive.size.y &&
               fabs(intersection.z - primitive.p0.z) < primitive.size.z;
            if( m_hMaterials[primitive.materialId.x].innerIllumination.x != 0.f )
            {
               // Chessboard like Lights
               collision &= int(fabs(intersection.z))%4000<2000 && int(fabs(intersection.y))%4000<2000;
            }
            if( m_hMaterials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
               collision &= wireFrameMapping(intersection.y, intersection.z, m_hMaterials[primitive.materialId.x].attributes.w, primitive );
         }
         break;
      }
   case ptXYPlane:
   case ptCamera:
      {
         float z = ray.origin.z-primitive.p0.z;
         if( reverted*ray.origin.z<0.f && reverted*ray.origin.z>reverted*primitive.p0.z) 
         {
            intersection.z = primitive.p0.z;
            intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
            intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.y - primitive.p0.y) < primitive.size.y;
            if( m_hMaterials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
               collision &= wireFrameMapping(intersection.x, intersection.y, m_hMaterials[primitive.materialId.x].attributes.w, primitive );
         }
         if( !collision && reverted*ray.origin.z>0.f && reverted*ray.origin.z<reverted*primitive.p0.z )
         {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal.z = -normal.z;
            intersection.z = primitive.p0.z;
            intersection.x = ray.origin.x+z*ray.direction.x/-ray.direction.z;
            intersection.y = ray.origin.y+z*ray.direction.y/-ray.direction.z;
            collision = 
               fabs(intersection.x - primitive.p0.x) < primitive.size.x &&
               fabs(intersection.y - primitive.p0.y) < primitive.size.y;
            if( m_hMaterials[primitive.materialId.x].attributes.z == 2 ) // Wireframe
               collision &= wireFrameMapping(intersection.x, intersection.y, m_hMaterials[primitive.materialId.x].attributes.w, primitive );
         }
         break;
      }
   }

   if( collision ) 
   {
      // Shadow intensity
      shadowIntensity = 1.f; //m_sceneInfo.shadowIntensity.x*(1.f-m_hMaterials[primitive.materialId.x].transparency.x);

      float4 color = m_hMaterials[primitive.materialId.x].color;
      if( primitive.type.x == ptCamera || m_hMaterials[primitive.materialId.x].textureMapping.z != TEXTURE_NONE )
      {
         color = cubeMapping(primitive, intersection );
         shadowIntensity = color.w;
      }

      if( (color.x+color.y+color.z)/3.f >= m_sceneInfo.transparentColor.x ) 
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
bool CPUKernel::triangleIntersection( 
   const Primitive& triangle, 
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
   float3 E01={triangle.p1.x-triangle.p0.x,triangle.p1.y-triangle.p0.y,triangle.p1.z-triangle.p0.z};
   float3 E03={triangle.p2.x-triangle.p0.x,triangle.p2.y-triangle.p0.y,triangle.p2.z-triangle.p0.z};
   float3 P = crossProduct(ray.direction,E03);
   float det = dot(E01,P);
   
   if (fabs(det) < EPSILON) return false;
   
   float3 T = {ray.origin.x-triangle.p0.x,ray.origin.y-triangle.p0.y,ray.origin.z-triangle.p0.z};
   float a = dot(T,P)/det;
   if (a < 0.f || a > 1.f) return false;

   float3 Q = crossProduct(T,E01);
   float b = dot(ray.direction,Q)/det;
   if (b < 0.f || b > 1.f) return false;

   // Reject rays using the barycentric coordinates of
   // the intersection point with respect to T′.
   if ((a+b) > 1.f) 
   {
      float3 E23 = {triangle.p0.x-triangle.p1.x,triangle.p0.y-triangle.p1.y,triangle.p0.z-triangle.p1.z};
      float3 E21 = {triangle.p1.x-triangle.p1.x,triangle.p1.y-triangle.p1.y,triangle.p1.z-triangle.p1.z};
      float3 P_ = crossProduct(ray.direction,E21);
      float det_ = dot(E23,P_);
      if(fabs(det_) < EPSILON) return false;
      float3 T_ = {ray.origin.x-triangle.p2.x,ray.origin.y-triangle.p2.y,ray.origin.z-triangle.p2.z};
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
   intersection.x = ray.origin.x+t*ray.direction.x;
   intersection.y = ray.origin.y+t*ray.direction.y;
   intersection.z = ray.origin.z+t*ray.direction.z;

   // Normal
   normal = triangle.n0;
   float3 v0={triangle.p0.x-intersection.x,triangle.p0.y-intersection.y,triangle.p0.z-intersection.z};
   float3 v1={triangle.p1.x-intersection.x,triangle.p1.y-intersection.y,triangle.p1.z-intersection.z};
   float3 v2={triangle.p2.x-intersection.x,triangle.p2.y-intersection.y,triangle.p2.z-intersection.z};

   areas.x = 0.5f*vectorLength(crossProduct( v1,v2 ));
   areas.y = 0.5f*vectorLength(crossProduct( v0,v2 ));
   areas.z = 0.5f*vectorLength(crossProduct( v0,v1 ));
   
   normal.x=(triangle.n0.x*areas.x + triangle.n1.x*areas.y + triangle.n2.x*areas.z)/(areas.x+areas.y+areas.z);
   normal.y=(triangle.n0.y*areas.x + triangle.n1.y*areas.y + triangle.n2.y*areas.z)/(areas.x+areas.y+areas.z);
   normal.z=(triangle.n0.z*areas.x + triangle.n1.z*areas.y + triangle.n2.z*areas.z)/(areas.x+areas.y+areas.z);
   normal=normalize(normal);

   float3 dir = normalize(ray.direction);
   float r = dot(dir,normal);

   if( r>0.f )
   {
      normal.x *= -1.f;
      normal.y *= -1.f;
      normal.z *= -1.f;
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
bool CPUKernel::intersectionWithPrimitives(
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
	float minDistance  = m_sceneInfo.viewDistance.x/iteration;

   Ray r;
   r.origin    = ray.origin;
   r.direction.x = ray.direction.x-ray.origin.x;
   r.direction.y = ray.direction.y-ray.origin.y;
   r.direction.z = ray.direction.z-ray.origin.z;
   computeRayAttributes( r );

   float3 intersection = {0.f,0.f,0.f};
   float3 normal       = {0.f,0.f,0.f};
   bool i = false;
   float shadowIntensity = 0.f;

   int cptBoxes = 0;
   while(cptBoxes<m_nbActiveBoxes[m_frame])
	{
      BoundingBox& box = m_hBoundingBoxes[cptBoxes];
      if( boxIntersection(box, r, 0.f, minDistance) )
      {
         // Intersection with Box
         if( m_sceneInfo.renderBoxes.x!=0 )
         {
            colorBox.x += m_hMaterials[box.startIndex.x%NB_MAX_MATERIALS].color.x/50.f;
            colorBox.y += m_hMaterials[box.startIndex.x%NB_MAX_MATERIALS].color.y/50.f;
            colorBox.z += m_hMaterials[box.startIndex.x%NB_MAX_MATERIALS].color.z/50.f;
         }
         else
         {
            // Intersection with primitive within boxes
            for( int cptPrimitives=0; cptPrimitives<box.nbPrimitives.x; ++cptPrimitives )
            { 
               Primitive& primitive = m_hPrimitives[box.startIndex.x+cptPrimitives];
               Material& material = m_hMaterials[primitive.materialId.x];
               if( material.attributes.x==0 || (material.attributes.x==1 && currentMaterialId != primitive.materialId.x)) // !!!! TEST SHALL BE REMOVED TO INCREASE TRANSPARENCY QUALITY !!!
               {
                  float3 areas = {0.f,0.f,0.f};
                  i = false;
                  switch( primitive.type.x )
                  {
                  case ptEnvironment :
                  case ptSphere:
                     {
                        i = sphereIntersection  ( primitive, r, intersection, normal, shadowIntensity, back ); 
                        break;
                     }
                  case ptCylinder: 
                     {
                        i = cylinderIntersection( primitive, r, intersection, normal, shadowIntensity, back ); 
                        break;
                     }
                  case ptEllipsoid:
                     {
                        i = ellipsoidIntersection( primitive, r, intersection, normal, shadowIntensity, back );
                        break;
                     }
                  case ptTriangle:
                     {
                        back = false;
                        i = triangleIntersection( primitive, r, intersection, normal, areas, shadowIntensity, back ); 
                        break;
                     }
                  default: 
                     {
                        back = false;
                        i = planeIntersection   ( primitive, r, intersection, normal, shadowIntensity, false); 
                        break;
                     }
                  }

                  float3 d;
                  d.x = intersection.x-r.origin.x;
                  d.y = intersection.y-r.origin.y;
                  d.z = intersection.z-r.origin.z;
                  float distance = vectorLength(d);
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

Convert float3 into OpenGL RGB color
________________________________________________________________________________
*/
void CPUKernel::makeColor(
   float4&   color,
   int       index)
{
   int mdc_index = index*gColorDepth; 
   color.x = (color.x>1.f) ? 1.f : color.x;
   color.y = (color.y>1.f) ? 1.f : color.y; 
   color.z = (color.z>1.f) ? 1.f : color.z;

   switch( m_sceneInfo.misc.x )
   {
   case otOpenGL: 
      {
         // OpenGL
         m_bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
         m_bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
         m_bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
         break;
      }
   case otDelphi: 
      {
         // Delphi
         m_bitmap[mdc_index  ] = (char)(color.z*255.f); // Blue
         m_bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
         m_bitmap[mdc_index+2] = (char)(color.x*255.f); // Red
         break;
      }
   case otJPEG: 
      {
         mdc_index = (m_sceneInfo.width.x*m_sceneInfo.height.x-index)*gColorDepth; 
         // JPEG
         m_bitmap[mdc_index+2] = (char)(color.z*255.f); // Blue
         m_bitmap[mdc_index+1] = (char)(color.y*255.f); // Green
         m_bitmap[mdc_index  ] = (char)(color.x*255.f); // Red
         break;
      }
   }
}

/*
________________________________________________________________________________

Shadows computation
We do not consider the object from which the ray is launched...
This object cannot shadow itself !

We now have to find the intersection between the considered object and the ray 
which m_viewPos is the considered 3D float3 and which direction is defined by the 
light source center.
.
. * Lamp                     Ray = m_viewPos -> Light Source Center
.  \
.   \##
.   #### object
.    ##
.      \
.       \  m_viewPos
.--------O-------
.
@return 1.f when pixel is in the shades

________________________________________________________________________________
*/
float CPUKernel::processShadows(
   const float3& lampCenter, 
   const float3& m_viewPos, 
   const int&    objectId,
   const int&    iteration,
   float4&       color)
{
   float result = 0.f;
   int cptBoxes = 0;
   color.x = 0.f;
   color.y = 0.f;
   color.z = 0.f;
   int it=-1;
   Ray r;
   r.origin    = m_viewPos;
   r.direction.x = lampCenter.x-m_viewPos.x;
   r.direction.y = lampCenter.y-m_viewPos.y;
   r.direction.z = lampCenter.z-m_viewPos.z;
   computeRayAttributes( r );

   while( result<m_sceneInfo.shadowIntensity.x && cptBoxes<m_nbActiveBoxes[m_frame] )
   {

      BoundingBox& box = m_hBoundingBoxes[cptBoxes];
      if( boxIntersection(box, r, 0.f, m_sceneInfo.viewDistance.x))
      {
         int cptPrimitives = 0;
         while( result<m_sceneInfo.shadowIntensity.x && cptPrimitives<box.nbPrimitives.x)
         {
            float3 intersection = {0.f,0.f,0.f};
            float3 normal       = {0.f,0.f,0.f};
            float3 areas        = {0.f,0.f,0.f};
            float  shadowIntensity = 0.f;

            Primitive& primitive = m_hPrimitives[box.startIndex.x+cptPrimitives];
            if( primitive.index.x!=objectId && m_hMaterials[primitive.materialId.x].attributes.x==0)
            {

               bool hit = false;
               bool back;
               switch(primitive.type.x)
               {
               case ptSphere   : hit=sphereIntersection   ( primitive, r, intersection, normal, shadowIntensity, back ); break;
               case ptEllipsoid: hit=ellipsoidIntersection( primitive, r, intersection, normal, shadowIntensity, back ); break;
               case ptCylinder :	hit=cylinderIntersection ( primitive, r, intersection, normal, shadowIntensity, back ); break;
               case ptTriangle :	hit=triangleIntersection ( primitive, r, intersection, normal, areas, shadowIntensity, back ); break;
               case ptCamera   : hit=false; break;
               default         : hit=planeIntersection    ( primitive, r, intersection, normal, shadowIntensity, false ); break;
               }

               if( hit )
               {
                  float3 O_I;
                  O_I.x = intersection.x-r.origin.x;
                  O_I.y = intersection.y-r.origin.y;
                  O_I.z = intersection.z-r.origin.z;

                  float3 O_L;
                  O_L.x = r.direction.x;
                  O_L.y = r.direction.y;
                  O_L.z = r.direction.z;

                  float l = vectorLength(O_I);
                  if( l>EPSILON && l<vectorLength(O_L) )
                  {
                     float ratio = shadowIntensity*m_sceneInfo.shadowIntensity.x;
                     if( m_hMaterials[primitive.materialId.x].transparency.x != 0.f )
                     {
                        O_L=normalize(O_L);
                        float a=fabs(dot(O_L,normal));
                        float r = (m_hMaterials[primitive.materialId.x].transparency.x == 0.f ) ? 1.f : (1.f-0.8f*m_hMaterials[primitive.materialId.x].transparency.x);
                        ratio *= r*a;
                        // Shadow color
                        color.x  += ratio*(0.3f-0.3f*m_hMaterials[primitive.materialId.x].color.x);
                        color.y  += ratio*(0.3f-0.3f*m_hMaterials[primitive.materialId.x].color.y);
                        color.z  += ratio*(0.3f-0.3f*m_hMaterials[primitive.materialId.x].color.z);
                     }
                     result += ratio;
                  }
                  it++;
               }
            }
            cptPrimitives++;
         }
      }
      cptBoxes++;
   }
   result = (result>m_sceneInfo.shadowIntensity.x) ? m_sceneInfo.shadowIntensity.x : result;
   result = (result<0.f) ? 0.f : result;
   return result;
}

/*
________________________________________________________________________________

Intersection Shader
________________________________________________________________________________
*/
float4 CPUKernel::intersectionShader( 
   const Primitive& primitive, 
   const float3&    intersection,
   const float3&    areas)
{
   float4 colorAtIntersection = m_hMaterials[primitive.materialId.x].color;
   colorAtIntersection.w = 0.f; // w attribute is used to dtermine light intensity of the material

   switch( primitive.type.x ) 
   {
   case ptCylinder:
      {
         if(m_hMaterials[primitive.materialId.x].textureMapping.z != TEXTURE_NONE)
         {
            colorAtIntersection = sphereUVMapping(primitive, intersection );
         }
         break;
      }
   case ptEnvironment:
   case ptSphere:
   case ptEllipsoid:
      {
         if(m_hMaterials[primitive.materialId.x].textureMapping.z != TEXTURE_NONE)
         {
            colorAtIntersection = sphereUVMapping(primitive, intersection );
         }
         break;
      }
   case ptCheckboard :
      {
         if( m_hMaterials[primitive.materialId.x].textureMapping.z != TEXTURE_NONE ) 
         {
            colorAtIntersection = cubeMapping( primitive, intersection );
         }
         else 
         {
            int x = static_cast<int>(m_sceneInfo.viewDistance.x + ((intersection.x - primitive.p0.x)/primitive.size.x));
            int z = static_cast<int>(m_sceneInfo.viewDistance.x + ((intersection.z - primitive.p0.z)/primitive.size.x));
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
         if( m_hMaterials[primitive.materialId.x].textureMapping.z != TEXTURE_NONE ) 
         {
            colorAtIntersection = cubeMapping( primitive, intersection );
         }
         break;
      }
   case ptTriangle:
      {
         if( m_hMaterials[primitive.materialId.x].textureMapping.z != TEXTURE_NONE ) 
         {
            colorAtIntersection = triangleUVMapping( primitive, intersection, areas );
         }
         break;
      }
   }
   return colorAtIntersection;
}

/*
________________________________________________________________________________

Primitive shader
________________________________________________________________________________
*/
float4 CPUKernel::primitiveShader(
   const float3& m_viewPos,
   const float3& normal, 
   const int&    objectId, 
   const float3& intersection,
   const float3& areas,
   const int&    iteration,
   float4&       refractionFromColor,
   float&        shadowIntensity,
   float4&       totalBlinn)
{
   Primitive primitive = m_hPrimitives[objectId];
   float4 color = m_hMaterials[primitive.materialId.x].color;
   float4 lampsColor = { 0.f, 0.f, 0.f, 0.f };

   // Lamp Impact
   shadowIntensity=0.f;

   if( m_hMaterials[primitive.materialId.x].innerIllumination.x!=0.f || 
       m_hMaterials[primitive.materialId.x].attributes.z==2 )
   {
      // Wireframe returns constant color
      return color; 
   }

   if( m_hMaterials[primitive.materialId.x].attributes.z == 1 )
   {
      // Sky box returns color with constant lightning
      return intersectionShader( primitive, intersection, areas );
   }

   if( m_sceneInfo.graphicsLevel.x>0 )
   {
      color.x *= m_hMaterials[primitive.materialId.x].innerIllumination.x;
      color.y *= m_hMaterials[primitive.materialId.x].innerIllumination.x;
      color.z *= m_hMaterials[primitive.materialId.x].innerIllumination.x;
      for( int cpt=0; cpt<m_lightInformationSize; ++cpt ) 
      {
         int cptLamp = cpt;
         if(m_lightInformation[cptLamp].attribute.x != primitive.index.x)
         {
            float3 center;
            // randomize lamp center
            center.x = m_lightInformation[cptLamp].location.x;
            center.y = m_lightInformation[cptLamp].location.y;
            center.z = m_lightInformation[cptLamp].location.z;

            if( m_lightInformation[cptLamp].attribute.x>=0 && 
                m_lightInformation[cptLamp].attribute.x<m_nbActivePrimitives[m_frame])
            {
               Primitive& lamp = m_hPrimitives[m_lightInformation[cptLamp].attribute.x];
               int t = 3*m_sceneInfo.misc.y;
               t = t%(m_sceneInfo.width.x*m_sceneInfo.height.x-3);
               center.x += m_hMaterials[lamp.materialId.x].innerIllumination.y*m_hRandoms[t  ]*m_sceneInfo.pathTracingIteration.x/float(m_sceneInfo.maxPathTracingIterations.x);
               center.y += m_hMaterials[lamp.materialId.x].innerIllumination.y*m_hRandoms[t+1]*m_sceneInfo.pathTracingIteration.x/float(m_sceneInfo.maxPathTracingIterations.x);
               center.z += m_hMaterials[lamp.materialId.x].innerIllumination.y*m_hRandoms[t+2]*m_sceneInfo.pathTracingIteration.x/float(m_sceneInfo.maxPathTracingIterations.x);
            }

            float4 shadowColor = {0.f,0.f,0.f,0.f};
            if( m_sceneInfo.graphicsLevel.x>3 &&
                iteration<4 && 
                m_hMaterials[primitive.materialId.x].innerIllumination.x==0.f ) 
            {
               shadowIntensity = processShadows(
                  center, intersection, m_lightInformation[cptLamp].attribute.x, iteration, shadowColor );
            }

            if( m_sceneInfo.graphicsLevel.x>0 )
            {
               float3 lightRay;
               lightRay.x = center.x - intersection.x;
               lightRay.y = center.y - intersection.y;
               lightRay.z = center.z - intersection.z;

               lightRay = normalize(lightRay);
               // --------------------------------------------------------------------------------
               // Lambert
               // --------------------------------------------------------------------------------
               float lambert = (m_postProcessingInfo.type.x==ppe_ambientOcclusion) ? 0.6f : dot(normal,lightRay);
               // Transparent m_hMaterials are lighted on both sides but the amount of light received by the "dark side" 
               // depends on the transparency rate.
               lambert *= (lambert<0.f) ? -m_hMaterials[primitive.materialId.x].transparency.x : lambert;
               lambert *= m_lightInformation[cptLamp].color.w;
               lambert *= (1.f-shadowIntensity);

               // Lighted object, not in the shades

               lampsColor.x += lambert*m_lightInformation[cptLamp].color.x*m_lightInformation[cptLamp].color.w - shadowColor.x;
               lampsColor.y += lambert*m_lightInformation[cptLamp].color.y*m_lightInformation[cptLamp].color.w - shadowColor.y;
               lampsColor.z += lambert*m_lightInformation[cptLamp].color.z*m_lightInformation[cptLamp].color.w - shadowColor.z;

               if( m_sceneInfo.graphicsLevel.x>1 && shadowIntensity<m_sceneInfo.shadowIntensity.x )
               {
                  // --------------------------------------------------------------------------------
                  // Blinn - Phong
                  // --------------------------------------------------------------------------------
                  float3 viewRay;
                  viewRay.x = intersection.x - m_viewPos.x;
                  viewRay.y = intersection.y - m_viewPos.y;
                  viewRay.z = intersection.z - m_viewPos.z;
                  viewRay = normalize(viewRay);

                  float3 blinnDir;
                  blinnDir.x = lightRay.x - viewRay.x;
                  blinnDir.y = lightRay.y - viewRay.y;
                  blinnDir.z = lightRay.z - viewRay.z;

                  float temp = sqrt(dot(blinnDir,blinnDir));
                  if (temp != 0.f ) 
                  {
                     // Specular reflection
                     blinnDir.x = (1.f / temp) * blinnDir.x;
                     blinnDir.y = (1.f / temp) * blinnDir.y;
                     blinnDir.z = (1.f / temp) * blinnDir.z;

                     float blinnTerm = dot(blinnDir,normal);
                     blinnTerm = ( blinnTerm < 0.f) ? 0.f : blinnTerm;

                     blinnTerm = m_hMaterials[primitive.materialId.x].specular.x * pow(blinnTerm,m_hMaterials[primitive.materialId.x].specular.y);

                     totalBlinn.x += m_lightInformation[cptLamp].color.x * m_lightInformation[cptLamp].color.w * blinnTerm;
                     totalBlinn.y += m_lightInformation[cptLamp].color.y * m_lightInformation[cptLamp].color.w * blinnTerm;
                     totalBlinn.z += m_lightInformation[cptLamp].color.z * m_lightInformation[cptLamp].color.w * blinnTerm;
                  }
               }
            }
         }

         // Final color
         float4 intersectionColor = intersectionShader( primitive, intersection, areas );

         // Light impact on material
         color.x += intersectionColor.x*lampsColor.x;
         color.y += intersectionColor.y*lampsColor.y;
         color.z += intersectionColor.z*lampsColor.z;

         // Saturate color
         saturateVector(color);

         refractionFromColor = intersectionColor; // Refraction depending on color;
         saturateVector( totalBlinn );
      }
   }
   return color;
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
intersection float3 and  will also compute the shadows. 
The resulted color is stored in result.                     
The first parameter is the closest object to the intersection (following 
the ray). It can  be considered as a light source if its inner light rate 
is > 0.                            
________________________________________________________________________________
*/
float4 CPUKernel::launchRay( 
   const Ray&       ray, 
   float3&          intersection,
   float&           depthOfField,
   int4&            primitiveXYId)
{
   float4 intersectionColor   = {0.f,0.f,0.f,0.f};

   float3 closestIntersection = {0.f,0.f,0.f};
   float3 firstIntersection   = {0.f,0.f,0.f};
   float3 normal              = {0.f,0.f,0.f};
   int    closestPrimitive  = 0;
   bool   carryon           = true;
   Ray    rayOrigin         = ray;
   float  initialRefraction = 1.f;
   int    iteration         = 0;
   primitiveXYId.x = -1;
   int currentMaterialId=-2;

   // TODO
   float  colorContributions[NB_MAX_ITERATIONS];
   float4 colors[NB_MAX_ITERATIONS];
   memset(&colorContributions[0],0,sizeof(float)*NB_MAX_ITERATIONS);
   memset(&colors[0],0,sizeof(float4)*NB_MAX_ITERATIONS);

   float4 recursiveBlinn = { 0.f, 0.f, 0.f, 0.f };

   // Variable declarations
   float  shadowIntensity = 0.f;
   float4 refractionFromColor;
   float3 reflectedTarget;
   float4 colorBox = {0.f,0.f,0.f,0.f};
   bool   back = false;

#ifdef PHOTON_ENERGY
   // Photon energy
   float photonDistance = m_sceneInfo.viewDistance.x;
   float previousTransparency = 1.f;
#endif // PHOTON_ENERGY

   // Reflected rays
   int reflectedRays=-1;
   Ray reflectedRay;
   float reflectedRatio;

   float4 rBlinn = {0.f,0.f,0.f,0.f};
   int currentMaxIteration = ( m_sceneInfo.graphicsLevel.x<3 ) ? 1 : m_sceneInfo.nbRayIterations.x+m_sceneInfo.pathTracingIteration.x;
   currentMaxIteration = (currentMaxIteration>NB_MAX_ITERATIONS) ? NB_MAX_ITERATIONS : currentMaxIteration;
#ifdef PHOTON_ENERGY
   while( iteration<currentMaxIteration && carryon && photonDistance>0.f ) 
#else
   while( iteration<currentMaxIteration && carryon ) 
#endif // PHOTON_ENERGY
   {
      float3 areas = {0.f,0.f,0.f};
      // If no intersection with lamps detected. Now compute intersection with Primitives
      if( carryon ) 
      {
         carryon = intersectionWithPrimitives(
            rayOrigin, iteration,  closestPrimitive, closestIntersection, 
            normal, areas, colorBox, back, currentMaterialId);
      }

      if( carryon ) 
      {
         currentMaterialId = m_hPrimitives[closestPrimitive].materialId.x;

         if ( iteration==0 )
         {
            colors[iteration].x = 0.f;
            colors[iteration].y = 0.f;
            colors[iteration].z = 0.f;
            colors[iteration].w = 0.f;
            colorContributions[iteration]=1.f;

            firstIntersection = closestIntersection;
            primitiveXYId.x = m_hPrimitives[closestPrimitive].index.x;
         }

#ifdef PHOTON_ENERGY
         // Photon
         float3 d;
         d.x = closestIntersection.x-rayOrigin.origin.x;
         d.y = closestIntersection.y-rayOrigin.origin.y;
         d.z = closestIntersection.z-rayOrigin.origin.z;
         photonDistance -= vectorLength(d) * (2.f-previousTransparency);
         previousTransparency = back ? 1.f : m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].transparency.x;
#endif // PHOTON_ENERGY

         // Get object color
         colors[iteration] =
            primitiveShader( 
            rayOrigin.origin, normal, 
            closestPrimitive, closestIntersection, areas, 
            iteration, refractionFromColor, shadowIntensity, rBlinn );

         // ----------
         // Refraction
         // ----------

         if( m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].transparency.x != 0.f ) 
         {
            // Replace the normal using the intersection color
            // r,g,b become x,y,z... What the fuck!!
            if( m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].textureMapping.z != TEXTURE_NONE) 
            {
               normal.x *= (colors[iteration].x-0.5f);
               normal.y *= (colors[iteration].y-0.5f);
               normal.z *= (colors[iteration].z-0.5f);
            }

            // Back of the object? If so, reset refraction to 1.f (air)
            float refraction = back ? 1.f : m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].refraction.x;

            // Actual refraction
            float3 O_E;
            O_E.x = rayOrigin.origin.x - closestIntersection.x;
            O_E.y = rayOrigin.origin.y - closestIntersection.y;
            O_E.z = rayOrigin.origin.z - closestIntersection.z;
            O_E = normalize(O_E);
            vectorRefraction( rayOrigin.direction, O_E, refraction, normal, initialRefraction );

            reflectedTarget.x = closestIntersection.x - rayOrigin.direction.x;
            reflectedTarget.y = closestIntersection.y - rayOrigin.direction.y;
            reflectedTarget.z = closestIntersection.z - rayOrigin.direction.z;

            colorContributions[iteration] = m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].transparency.x;

            // Prepare next ray
            initialRefraction = refraction;

            if( reflectedRays==-1 && m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].reflection.x != 0.f )
            {
               vectorReflection( reflectedRay.direction, O_E, normal );
               float3 rt;
               rt.x = closestIntersection.x - reflectedRay.direction.x;
               rt.y = closestIntersection.y - reflectedRay.direction.y;
               rt.z = closestIntersection.z - reflectedRay.direction.z;

               reflectedRay.origin.x = closestIntersection.x + rt.x*0.00001f;
               reflectedRay.origin.y = closestIntersection.y + rt.y*0.00001f;
               reflectedRay.origin.z = closestIntersection.z + rt.z*0.00001f;

               reflectedRay.direction = rt;
               reflectedRatio = m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].reflection.x;
               reflectedRays=iteration;
            }
         }
         else
         {
            // ----------
            // Reflection
            // ----------
            if( m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].reflection.x != 0.f ) 
            {
               float3 O_E;
               O_E.x = rayOrigin.origin.x - closestIntersection.x;
               O_E.y = rayOrigin.origin.y - closestIntersection.y;
               O_E.z = rayOrigin.origin.z - closestIntersection.z;
               vectorReflection( rayOrigin.direction, O_E, normal );

               reflectedTarget.x = closestIntersection.x - rayOrigin.direction.x;
               reflectedTarget.y = closestIntersection.y - rayOrigin.direction.y;
               reflectedTarget.z = closestIntersection.z - rayOrigin.direction.z;
               colorContributions[iteration] = m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].reflection.x;
            }
            else 
            {
               colorContributions[iteration] = 1.f;
               carryon = false;
            }         
         }

         // Contribute to final color
         recursiveBlinn.x += rBlinn.x;
         recursiveBlinn.y += rBlinn.y;
         recursiveBlinn.z += rBlinn.z;

         rayOrigin.origin.x = closestIntersection.x + reflectedTarget.x*0.00001f; 
         rayOrigin.origin.y = closestIntersection.y + reflectedTarget.y*0.00001f; 
         rayOrigin.origin.z = closestIntersection.z + reflectedTarget.z*0.00001f; 
         rayOrigin.direction = reflectedTarget;

         // Noise management
         if( m_sceneInfo.pathTracingIteration.x != 0 && m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].color.w != 0.f)
         {
            // Randomize view
            float ratio = m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].color.w;
            ratio *= (m_hMaterials[m_hPrimitives[closestPrimitive].materialId.x].transparency.x==0.f) ? 1000.f : 1.f;
            int rindex = 3*m_sceneInfo.misc.y + m_sceneInfo.pathTracingIteration.x;
            rindex = rindex%(m_sceneInfo.width.x*m_sceneInfo.height.x);
            rayOrigin.direction.x += m_hRandoms[rindex  ]*ratio;
            rayOrigin.direction.y += m_hRandoms[rindex+1]*ratio;
            rayOrigin.direction.z += m_hRandoms[rindex+2]*ratio;
         }
      }
      else
      {
#ifdef GRADIANT_BACKGROUND
         // Background
         float3 normal = {0.f, 1.f, 0.f };
         float3 dir;
         dir.x = rayOrigin.direction.x-rayOrigin.origin.x;
         dir.y = rayOrigin.direction.y-rayOrigin.origin.y;
         dir.z = rayOrigin.direction.z-rayOrigin.origin.z;
         dir = normalize(dir);
         float angle = 0.5f*fabs(dot( normal, dir));
         angle = (angle>1.f) ? 1.f: angle;
         colors[iteration].x = (1.f-angle)*m_sceneInfo.backgroundColor.x;
         colors[iteration].y = (1.f-angle)*m_sceneInfo.backgroundColor.y;
         colors[iteration].z = (1.f-angle)*m_sceneInfo.backgroundColor.z;
#else
         colors[iteration] = m_sceneInfo.backgroundColor;
#endif // GRADIANT_BACKGROUND
         colorContributions[iteration] = 1.f;
      }
      iteration++;
   }

   if( m_sceneInfo.graphicsLevel.x>=3 && reflectedRays != -1 ) // TODO: Draft mode should only test "m_sceneInfo.pathTracingIteration.x==iteration"
   {
      float3 areas = {0.f,0.f,0.f};
      // TODO: Dodgy implementation		
      if( intersectionWithPrimitives(
         reflectedRay,
         reflectedRays,  
         closestPrimitive, closestIntersection, 
         normal, areas, colorBox, back, currentMaterialId) )
      {
         float4 color = primitiveShader( 
            reflectedRay.origin, normal, closestPrimitive, 
            closestIntersection, areas, 
            reflectedRays, 
            refractionFromColor, shadowIntensity, rBlinn );

         colors[reflectedRays].x += color.x*reflectedRatio;
         colors[reflectedRays].y += color.y*reflectedRatio;
         colors[reflectedRays].z += color.z*reflectedRatio;
      }
   }

   for( int i=iteration-2; i>=0; --i)
   {
      colors[i].x = colors[i].x*(1.f-colorContributions[i]) + colors[i+1].x*colorContributions[i];
      colors[i].y = colors[i].y*(1.f-colorContributions[i]) + colors[i+1].y*colorContributions[i];
      colors[i].z = colors[i].z*(1.f-colorContributions[i]) + colors[i+1].z*colorContributions[i];
   }
   intersectionColor = colors[0];
   intersectionColor.x += recursiveBlinn.x;
   intersectionColor.y += recursiveBlinn.y;
   intersectionColor.z += recursiveBlinn.z;

   intersection = closestIntersection;

   Primitive& primitive=m_hPrimitives[closestPrimitive];
   float3 l;
   l.x = firstIntersection.x - ray.origin.x;
   l.y = firstIntersection.y - ray.origin.y;
   l.z = firstIntersection.z - ray.origin.z;
   float len = vectorLength(l);
   if( m_hMaterials[primitive.materialId.x].attributes.z == 0 ) // Wireframe
   {
#ifdef PHOTON_ENERGY
      // --------------------------------------------------
      // Photon energy
      // --------------------------------------------------
      intersectionColor.x *= ( photonDistance>0.f) ? (photonDistance/m_sceneInfo.viewDistance.x) : 0.f;
      intersectionColor.y *= ( photonDistance>0.f) ? (photonDistance/m_sceneInfo.viewDistance.x) : 0.f;
      intersectionColor.z *= ( photonDistance>0.f) ? (photonDistance/m_sceneInfo.viewDistance.x) : 0.f;
#endif // PHOTON_ENERGY

      // --------------------------------------------------
      // Fog
      // --------------------------------------------------
      //intersectionColor += m_hRandoms[((int)len + m_sceneInfo.misc.y)%100];

      // --------------------------------------------------
      // Background color
      // --------------------------------------------------
      float D1 = m_sceneInfo.viewDistance.x*0.95f;
      if( m_sceneInfo.misc.z==1 && len>D1)
      {
         float D2 = m_sceneInfo.viewDistance.x*0.05f;
         float a = len - D1;
         float b = 1.f-(a/D2);
         intersectionColor.x = intersectionColor.x*b + m_sceneInfo.backgroundColor.x*(1.f-b);
         intersectionColor.y = intersectionColor.y*b + m_sceneInfo.backgroundColor.y*(1.f-b);
         intersectionColor.z = intersectionColor.z*b + m_sceneInfo.backgroundColor.z*(1.f-b);
      }
   }
   depthOfField = (len-depthOfField)/m_sceneInfo.viewDistance.x;

   // Primitive information
   primitiveXYId.y = iteration;

   // Depth of field
   intersectionColor.x -= colorBox.x;
   intersectionColor.y -= colorBox.y;
   intersectionColor.z -= colorBox.z;
   saturateVector( intersectionColor );
   return intersectionColor;
}


/*
________________________________________________________________________________

Standard renderer
________________________________________________________________________________
*/
void CPUKernel::k_standardRenderer()
{
   int x;
#pragma omp parallel for
   for( x=0; x<m_sceneInfo.width.x; ++x )
   {
      for( int y=0; y<m_sceneInfo.height.x; ++y )
      {
         int index = y*m_sceneInfo.width.x+x;

         Ray ray;
         ray.origin = m_viewPos;
         ray.direction = m_viewDir;

         float3 rotationCenter = {0.f,0.f,0.f};
         bool antialiasingActivated = (m_sceneInfo.misc.w == 2);

         if( m_sceneInfo.pathTracingIteration.x == 0 )
         {
            m_postProcessingBuffer[index].x = 0.f;
            m_postProcessingBuffer[index].y = 0.f;
            m_postProcessingBuffer[index].z = 0.f;
            m_postProcessingBuffer[index].w = 0.f;
         }
         else
         {
		      int rindex = 3*(index+m_sceneInfo.misc.y) + 5000;
		      rindex = rindex%(m_sceneInfo.width.x*m_sceneInfo.height.x-3);
		      ray.origin.x += m_hRandoms[rindex  ]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		      ray.origin.y += m_hRandoms[rindex+1]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		      ray.origin.z += m_hRandoms[rindex+2]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);

            rindex = 3*(index+m_sceneInfo.misc.y);
		      rindex = rindex%(m_sceneInfo.width.x*m_sceneInfo.height.x-3);
		      ray.direction.x += m_hRandoms[rindex  ]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		      ray.direction.y += m_hRandoms[rindex+1]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		      ray.direction.z += m_hRandoms[rindex+2]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
         }

         float dof = m_postProcessingInfo.param1.x;
         float3 intersection;


         if( m_sceneInfo.misc.w == 1 ) // Isometric 3D
         {
            ray.direction.x = ray.origin.z*0.001f*(float)(x - (m_sceneInfo.width.x/2));
            ray.direction.y = -ray.origin.z*0.001f*(float)(y - (m_sceneInfo.height.x/2));
            ray.origin.x = ray.direction.x;
            ray.origin.y = ray.direction.y;
         }
         else
         {
            float ratio=(float)m_sceneInfo.width.x/(float)m_sceneInfo.height.x;
            float2 step;
            step.x=ratio*6400.f/(float)m_sceneInfo.width.x;
            step.y=6400.f/(float)m_sceneInfo.height.x;
            ray.direction.x = ray.direction.x - step.x*(float)(x - (m_sceneInfo.width.x/2));
            ray.direction.y = ray.direction.y + step.y*(float)(y - (m_sceneInfo.height.x/2));
         }

         vectorRotation( ray.origin, rotationCenter, m_angles );
         vectorRotation( ray.direction, rotationCenter, m_angles );

         // Antialisazing
         float2 AArotatedGrid[4] =
         {
            {  3.f,  5.f },
            {  5.f, -3.f },
            { -3.f, -5.f },
            { -5.f,  3.f }
         };

         if( m_sceneInfo.pathTracingIteration.x>m_hPrimitivesXYIds[index].y && 
            m_sceneInfo.pathTracingIteration.x>0 && 
            m_sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS ) break;

         float4 color = {0.f,0.f,0.f,0.f};
         if( antialiasingActivated )
         {
            Ray r=ray;
            for( int I=0; I<4; ++I )
            {
               r.direction.x = ray.origin.x + AArotatedGrid[I].x;
               r.direction.y = ray.origin.y + AArotatedGrid[I].y;
               float4 c = launchRay( r, intersection, dof, m_hPrimitivesXYIds[index]);
               color.x += c.x;
               color.y += c.y;
               color.z += c.z;
            }
         }
         float4 c = launchRay( ray, intersection, dof, m_hPrimitivesXYIds[index]);

         color.x += c.x;
         color.y += c.y;
         color.z += c.z;

         if( antialiasingActivated )
         {
            color.x /= 5.f;
            color.y /= 5.f;
            color.z /= 5.f;
         }

         if( m_sceneInfo.pathTracingIteration.x == 0 )
         {
            m_postProcessingBuffer[index].w = dof;
         }

         if( m_sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS )
         {
            m_postProcessingBuffer[index].x = color.x;
            m_postProcessingBuffer[index].y = color.y;
            m_postProcessingBuffer[index].z = color.z;
         }
         else
         {
            m_postProcessingBuffer[index].x += color.x;
            m_postProcessingBuffer[index].y += color.y;
            m_postProcessingBuffer[index].z += color.z;
         }
      }
   }
}

/*
________________________________________________________________________________

Standard renderer
________________________________________________________________________________
*/
void CPUKernel::k_fishEyeRenderer()
{
   int x;
#pragma omp parallel for
   for( x=0; x<m_sceneInfo.width.x; ++x )
   {
      for( int y=0; y<m_sceneInfo.height.x; ++y )
      {
         float4 color = {0.f,0.f,0.f,0.f};
         int index = y*m_sceneInfo.width.x+x;
         Ray ray;
         ray.origin = m_viewPos;
         ray.direction = m_viewDir;

         if( m_sceneInfo.pathTracingIteration.x == 0 )
         {
            m_postProcessingBuffer[index].x = 0.f;
            m_postProcessingBuffer[index].y = 0.f;
            m_postProcessingBuffer[index].z = 0.f;
            m_postProcessingBuffer[index].w = 0.f;
         }
         else
         {
		      int rindex = 3*(index+m_sceneInfo.misc.y) + 5000;
		      rindex = rindex%(m_sceneInfo.width.x*m_sceneInfo.height.x-3);
		      ray.origin.x += m_hRandoms[rindex  ]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		      ray.origin.y += m_hRandoms[rindex+1]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		      ray.origin.z += m_hRandoms[rindex+2]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);

            rindex = 3*(index+m_sceneInfo.misc.y);
		      rindex = rindex%(m_sceneInfo.width.x*m_sceneInfo.height.x-3);
		      ray.direction.x += m_hRandoms[rindex  ]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		      ray.direction.y += m_hRandoms[rindex+1]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
		      ray.direction.z += m_hRandoms[rindex+2]*m_postProcessingBuffer[index].w*m_postProcessingInfo.param1.x;//*float(sceneInfo.pathTracingIteration.x)/float(sceneInfo.maxPathTracingIterations.x);
         }

         float dof = m_postProcessingInfo.param1.x;
         float3 intersection;

         // Normal Y axis
         float2 step;
         step.y=6400.f/(float)m_sceneInfo.height.x;
         ray.origin.y = ray.origin.y + step.y*(float)(y - (m_sceneInfo.height.x/2));

         // 360° X axis
         step.x = 2.f*PI/static_cast<float>(m_sceneInfo.width.x);
         step.y = 2.f*PI/static_cast<float>(m_sceneInfo.height.x);

         float3 fishEyeAngles = {0.f,0.f,0.f};
         fishEyeAngles.y = m_angles.y + step.x*(float)x;
         //fishEyeAngles.x = m_angles.x + step.y*(float)y;

         vectorRotation( ray.direction, ray.origin, fishEyeAngles );

         //vectorRotation( ray.origin,    rotationCenter, m_angles );
         //vectorRotation( ray.direction, rotationCenter, m_angles );

         if( m_sceneInfo.pathTracingIteration.x>m_hPrimitivesXYIds[index].y && 
            m_sceneInfo.pathTracingIteration.x>0 && 
            m_sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS ) break;

         color = launchRay( ray, intersection, dof, m_hPrimitivesXYIds[index]);

         if( m_sceneInfo.pathTracingIteration.x == 0 )
         {
            m_postProcessingBuffer[index].w = dof;
         }

         if( m_sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS )
         {
            m_postProcessingBuffer[index].x = color.x;
            m_postProcessingBuffer[index].y = color.y;
            m_postProcessingBuffer[index].z = color.z;
         }
         else
         {
            m_postProcessingBuffer[index].x += color.x;
            m_postProcessingBuffer[index].y += color.y;
            m_postProcessingBuffer[index].z += color.z;
         }
      }
   }
}

/*
________________________________________________________________________________

Anaglyph Renderer
________________________________________________________________________________
*/
void CPUKernel::k_anaglyphRenderer()
{
   int x;
#pragma omp parallel for
   for( x=0; x<m_sceneInfo.width.x; ++x )
   {
      for( int y(0); y<m_sceneInfo.height.x; ++y )
      {
         int index = y*m_sceneInfo.width.x+x;
         float3 rotationCenter = {0.f,0.f,0.f};

         if( m_sceneInfo.pathTracingIteration.x == 0 )
         {
            m_postProcessingBuffer[index].x = 0.f;
            m_postProcessingBuffer[index].y = 0.f;
            m_postProcessingBuffer[index].z = 0.f;
            m_postProcessingBuffer[index].w = 0.f;
         }

         float dof = m_postProcessingInfo.param1.x;
         float3 intersection;
         Ray eyeRay;

         float ratio=(float)m_sceneInfo.width.x/(float)m_sceneInfo.height.x;
         float2 step;
         step.x=4.f*ratio*6400.f/(float)m_sceneInfo.width.x;
         step.y=4.f*6400.f/(float)m_sceneInfo.height.x;

         // Left eye
         eyeRay.origin.x = m_viewPos.x + m_sceneInfo.width3DVision.x;
         eyeRay.origin.y = m_viewPos.y;
         eyeRay.origin.z = m_viewPos.z;

         eyeRay.direction.x = m_viewDir.x - step.x*(float)(x - (m_sceneInfo.width.x/2));
         eyeRay.direction.y = m_viewDir.y + step.y*(float)(y - (m_sceneInfo.height.x/2));
         eyeRay.direction.z = m_viewDir.z;

         vectorRotation( eyeRay.origin, rotationCenter, m_angles );
         vectorRotation( eyeRay.direction, rotationCenter, m_angles );

         float4 colorLeft = launchRay( eyeRay, intersection, dof, m_hPrimitivesXYIds[index]);

         // Right eye
         eyeRay.origin.x = m_viewPos.x - m_sceneInfo.width3DVision.x;
         eyeRay.origin.y = m_viewPos.y;
         eyeRay.origin.z = m_viewPos.z;

         eyeRay.direction.x = m_viewDir.x - step.x*(float)(x - (m_sceneInfo.width.x/2));
         eyeRay.direction.y = m_viewDir.y + step.y*(float)(y - (m_sceneInfo.height.x/2));
         eyeRay.direction.z = m_viewDir.z;

         vectorRotation( eyeRay.origin, rotationCenter, m_angles );
         vectorRotation( eyeRay.direction, rotationCenter, m_angles );
         float4 colorRight = launchRay( eyeRay, intersection, dof, m_hPrimitivesXYIds[index] );

         float r1 = colorLeft.x*0.299f + colorLeft.y*0.587f + colorLeft.z*0.114f;
         float b1 = 0.f;
         float g1 = 0.f;

         float r2 = 0.f;
         float g2 = colorRight.y;
         float b2 = colorRight.z;

         if( m_sceneInfo.pathTracingIteration.x == 0 ) m_postProcessingBuffer[index].w = dof;
         if( m_sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS )
         {
            m_postProcessingBuffer[index].x = r1+r2;
            m_postProcessingBuffer[index].y = g1+g2;
            m_postProcessingBuffer[index].z = b1+b2;
         }
         else
         {
            m_postProcessingBuffer[index].x += r1+r2;
            m_postProcessingBuffer[index].y += g1+g2;
            m_postProcessingBuffer[index].z += b1+b2;
         }
      }
   }
}

/*
________________________________________________________________________________

3D Vision Renderer
________________________________________________________________________________
*/
void CPUKernel::k_3DVisionRenderer()
{
   int x;
#pragma omp parallel for
   for( x=0; x<m_sceneInfo.width.x; ++x )
   {
      for( int y(0); y<m_sceneInfo.height.x; ++y )
      {
         int index = y*m_sceneInfo.width.x+x;

         float3 rotationCenter = {0.f,0.f,0.f};

         if( m_sceneInfo.pathTracingIteration.x == 0 )
         {
            m_postProcessingBuffer[index].x = 0.f;
            m_postProcessingBuffer[index].y = 0.f;
            m_postProcessingBuffer[index].z = 0.f;
            m_postProcessingBuffer[index].w = 0.f;
         }

         float dof = m_postProcessingInfo.param1.x;
         float3 intersection;
         int halfWidth  = m_sceneInfo.width.x/2;

         float ratio=(float)m_sceneInfo.width.x/(float)m_sceneInfo.height.x;
         float2 step;
         step.x=ratio*6400.f/(float)m_sceneInfo.width.x;
         step.y=6400.f/(float)m_sceneInfo.height.x;

         Ray eyeRay;
         if( x<halfWidth ) 
         {
            // Left eye
            eyeRay.origin.x = m_viewPos.x + m_sceneInfo.width3DVision.x;
            eyeRay.origin.y = m_viewPos.y;
            eyeRay.origin.z = m_viewPos.z;

            eyeRay.direction.x = m_viewDir.x - step.x*(float)(x - (m_sceneInfo.width.x/2) + halfWidth/2 );
            eyeRay.direction.y = m_viewDir.y + step.y*(float)(y - (m_sceneInfo.height.x/2));
            eyeRay.direction.z = m_viewDir.z;
         }
         else
         {
            // Right eye
            eyeRay.origin.x = m_viewPos.x - m_sceneInfo.width3DVision.x;
            eyeRay.origin.y = m_viewPos.y;
            eyeRay.origin.z = m_viewPos.z;

            eyeRay.direction.x = m_viewDir.x - step.x*(float)(x - (m_sceneInfo.width.x/2) - halfWidth/2);
            eyeRay.direction.y = m_viewDir.y + step.y*(float)(y - (m_sceneInfo.height.x/2));
            eyeRay.direction.z = m_viewDir.z;
         }

         vectorRotation( eyeRay.origin, rotationCenter, m_angles );
         vectorRotation( eyeRay.direction, rotationCenter, m_angles );

         float4 color = launchRay(eyeRay, intersection, dof,m_hPrimitivesXYIds[index]);

         if( m_sceneInfo.pathTracingIteration.x == 0 ) m_postProcessingBuffer[index].w = dof;
         if( m_sceneInfo.pathTracingIteration.x<=NB_MAX_ITERATIONS )
         {
            m_postProcessingBuffer[index].x = color.x;
            m_postProcessingBuffer[index].y = color.y;
            m_postProcessingBuffer[index].z = color.z;
         }
         else
         {
            m_postProcessingBuffer[index].x += color.x;
            m_postProcessingBuffer[index].y += color.y;
            m_postProcessingBuffer[index].z += color.z;
         }
      }
   }
}

/*
________________________________________________________________________________

Post Processing Effect: Default
________________________________________________________________________________
*/
void CPUKernel::k_default() 
{
   int x;
#pragma omp parallel for
   for( x=0; x<m_sceneInfo.width.x; ++x )
   {
      for( int y(0); y<m_sceneInfo.height.x; ++y )
      {
         int index = y*m_sceneInfo.width.x+x;
         float4 localColor = m_postProcessingBuffer[index];

         if(m_sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS)
         {
            localColor.x /= (float)(m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
            localColor.y /= (float)(m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
            localColor.z /= (float)(m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
         }

         makeColor( localColor, index ); 
      }
   }
}

/*
________________________________________________________________________________

Post Processing Effect: Depth of field
________________________________________________________________________________
*/
void CPUKernel::k_depthOfField() 
{
   int x;
#pragma omp parallel for
   for( x=0; x<m_sceneInfo.width.x; ++x )
   {
      for( int y(0); y<m_sceneInfo.height.x; ++y )
      {
         int index = y*m_sceneInfo.width.x+x;
         float  depth = m_postProcessingInfo.param2.x*m_postProcessingBuffer[index].w;
         int    wh = m_sceneInfo.width.x*m_sceneInfo.height.x;

         float4 localColor = {0.f,0.f,0.f};
         for( int i=0; i<m_postProcessingInfo.param3.x; ++i )
         {
            int ix = i%wh;
            int iy = (i+m_sceneInfo.width.x)%wh;
            int xx = x+static_cast<int>(depth*m_hRandoms[ix]*0.5f);
            int yy = y+static_cast<int>(depth*m_hRandoms[iy]*0.5f);
            if( xx>=0 && xx<m_sceneInfo.width.x && yy>=0 && yy<m_sceneInfo.height.x )
            {
               int localIndex = yy*m_sceneInfo.width.x+xx;
               if( localIndex>=0 && localIndex<wh )
               {
                  localColor.x += m_postProcessingBuffer[localIndex].x;
                  localColor.y += m_postProcessingBuffer[localIndex].y;
                  localColor.z += m_postProcessingBuffer[localIndex].z;
               }
            }
            else
            {
               localColor.x += m_postProcessingBuffer[index].x;
               localColor.y += m_postProcessingBuffer[index].y;
               localColor.z += m_postProcessingBuffer[index].z;
            }
         }
         localColor.x /= m_postProcessingInfo.param3.x;
         localColor.y /= m_postProcessingInfo.param3.x;
         localColor.z /= m_postProcessingInfo.param3.x;

         if(m_sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS)
         {
            localColor.x /= (float)(m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
            localColor.y /= (float)(m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
            localColor.z /= (float)(m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
         }

         localColor.w = 1.f;

         makeColor( localColor, index ); 
      }
   }
}

/*
________________________________________________________________________________

Post Processing Effect: Ambiant Occlusion
________________________________________________________________________________
*/
void CPUKernel::k_ambiantOcclusion() 
{
   int x;
#pragma omp parallel for
   for( x=0; x<m_sceneInfo.width.x; ++x )
   {
      for( int y(0); y<m_sceneInfo.height.x; ++y )
      {
         int index = y*m_sceneInfo.width.x+x;
         float occ = 0.f;
         float4 localColor = m_postProcessingBuffer[index];
         float  depth = localColor.w;

         const int step = 16;
         for( int X=-step; X<step; X+=2 )
         {
            for( int Y=-step; Y<step; Y+=2 )
            {
               int xx = x+X;
               int yy = y+Y;
               if( xx>=0 && xx<m_sceneInfo.width.x && yy>=0 && yy<m_sceneInfo.height.x )
               {
                  int localIndex = yy*m_sceneInfo.width.x+xx;
                  if( m_postProcessingBuffer[localIndex].w>=depth)
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

         if(m_sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS)
         {
            localColor.x /= (float)(m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
            localColor.y /= (float)(m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
            localColor.z /= (float)(m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1);
         }

         saturateVector( localColor );
         localColor.w = 1.f;

         makeColor( localColor, index ); 
      }
   }
}

/*
________________________________________________________________________________

Post Processing Effect: Radiosity
________________________________________________________________________________
*/
void CPUKernel::k_radiosity()
{
   int x;
#pragma omp parallel for
   for( x=0; x<m_sceneInfo.width.x; ++x )
   {
      for( int y(0); y<m_sceneInfo.height.x; ++y )
      {
         int index = y*m_sceneInfo.width.x+x;
         int    wh = m_sceneInfo.width.x*m_sceneInfo.height.x;

         int div = (m_sceneInfo.pathTracingIteration.x>NB_MAX_ITERATIONS) ? (m_sceneInfo.pathTracingIteration.x-NB_MAX_ITERATIONS+1) : 1;

         float4 localColor = {0.f,0.f,0.f};
         for( int i=0; i<m_postProcessingInfo.param3.x; ++i )
         {
            int ix = (i+m_sceneInfo.pathTracingIteration.x)%wh;
            int iy = (i+m_sceneInfo.width.x)%wh;
            int xx = x+static_cast<int>(m_hRandoms[ix]*m_postProcessingInfo.param2.x/10.f);
            int yy = y+static_cast<int>(m_hRandoms[iy]*m_postProcessingInfo.param2.x/10.f);
            localColor.x += m_postProcessingBuffer[index].x;
            localColor.y += m_postProcessingBuffer[index].y;
            localColor.z += m_postProcessingBuffer[index].z;
            if( xx>=0 && xx<m_sceneInfo.width.x && yy>=0 && yy<m_sceneInfo.height.x )
            {
               int localIndex = yy*m_sceneInfo.width.x+xx;
               localColor.x += ( localIndex>=0 && localIndex<wh ) ? div*m_hPrimitivesXYIds[localIndex].z/255 : 0.f;
               localColor.y += ( localIndex>=0 && localIndex<wh ) ? div*m_hPrimitivesXYIds[localIndex].z/255 : 0.f;
               localColor.z += ( localIndex>=0 && localIndex<wh ) ? div*m_hPrimitivesXYIds[localIndex].z/255 : 0.f;
            }
         }
         localColor.x /= m_postProcessingInfo.param3.x;
         localColor.y /= m_postProcessingInfo.param3.x;
         localColor.z /= m_postProcessingInfo.param3.x;

         localColor.x /= div;
         localColor.y /= div;
         localColor.z /= div;

         localColor.w = 1.f;

         makeColor( localColor, index );
      }
   }
}

/*
________________________________________________________________________________

Post Processing Effect: Radiosity
________________________________________________________________________________
*/
void CPUKernel::k_oneColor()
{
}
/*
________________________________________________________________________________

Kernel launcher
________________________________________________________________________________
*/
void CPUKernel::render_begin( const float timer ) 
{
   GPUKernel::render_begin(timer);
   if( m_postProcessingBuffer==nullptr )
   {
      m_postProcessingBuffer = new float4[m_sceneInfo.width.x*m_sceneInfo.height.x];
   }

   switch( m_sceneInfo.renderingType.x ) 
   {
   case vtAnaglyph: k_anaglyphRenderer(); break;
   case vt3DVision: k_3DVisionRenderer(); break;
   case vtFishEye : k_fishEyeRenderer();  break;
   default        : k_standardRenderer(); break;
   }

   switch( m_postProcessingInfo.type.x )
   {
   case ppe_depthOfField    : k_depthOfField(); break;
   case ppe_ambientOcclusion: k_ambiantOcclusion(); break;
   case ppe_radiosity       : k_radiosity(); break;
   case ppe_oneColor        : k_oneColor(); break;
   default                  : k_default(); break;
   }
}

void CPUKernel::render_end()
{
   if( m_sceneInfo.misc.x == 0 )
   {
      ::glEnable(GL_TEXTURE_2D);
      ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      ::glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      ::glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
      ::glTexImage2D(GL_TEXTURE_2D, 0, 3, m_sceneInfo.width.x, m_sceneInfo.height.x, 0, GL_RGB, GL_UNSIGNED_BYTE, m_bitmap);
      ::glBegin(GL_QUADS);
      ::glTexCoord2f(1.f, 1.f);
      ::glVertex3f(-1.f, 1.f, 0.f);
      ::glTexCoord2f(0.f, 1.f);
      ::glVertex3f( 1.f, 1.f, 0.f);
      ::glTexCoord2f(0.f, 0.f);
      ::glVertex3f( 1.f,-1.f, 0.f);
      ::glTexCoord2f(1.f, 0.f);
      ::glVertex3f(-1.f,-1.f, 0.f);
      ::glEnd();
      ::glDisable(GL_TEXTURE_2D);
   }
}