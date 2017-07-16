#pragma once

#include "GPUKernel.h"

class RAYTRACINGENGINE_API CPUKernel : public GPUKernel
{
public:
   CPUKernel( bool activeLogging, int optimalNbOfPrimmitivesPerBox, int platform, int device );
   ~CPUKernel(void);

public:
	// ---------- Rendering ----------
	void render_begin( const float timer );
   void render_end();

protected:
   // Vectors
   void saturateVector( float4& v );
   float3 crossProduct( const float3& b, const float3& c );
   float vectorLength( const float3& v );
   float dot( const float3& v1, const float3& v2 );
   float3 normalize( const float3& v );
   void vectorReflection( float3& r, const float3& i, const float3& n );
   void vectorRefraction( float3& refracted, const float3 incident, const float n1, const float3 normal, const float n2 );
   void vectorRotation( float3& v, const float3& rotationCenter, const float3& angles );

protected:
   void computeRayAttributes(Ray& ray);

   // Texture mapping
   void juliaSet( const Primitive& primitive, const float x, const float y, float4& color );
   void mandelbrotSet( const Primitive& primitive, const float x, const float y, float4& color );
   float4 sphereUVMapping( const Primitive& primitive, const float3& intersection);
   float4 triangleUVMapping( const Primitive& primitive, const float3& intersection, const float3& areas);
   float4 cubeMapping( const Primitive& primitive, const float3& intersection);
   bool wireFrameMapping( float x, float y, int width, const Primitive& primitive );

protected:
   // Intersections
   bool boxIntersection( const BoundingBox& box, const Ray& ray, const float& t0, const float& t1 );
   bool ellipsoidIntersection( const Primitive& ellipsoid, const Ray& ray, float3& intersection, float3& normal, float& shadowIntensity, bool& back);
   bool sphereIntersection( const Primitive& sphere, const Ray& ray, float3& intersection, float3& normal, float& shadowIntensity, bool& back );
   bool cylinderIntersection( const Primitive& cylinder, const Ray& ray, float3& intersection, float3& normal, float& shadowIntensity, bool& back );
   bool planeIntersection( const Primitive& primitive, const Ray& ray, float3& intersection, float3& normal, float& shadowIntensity, bool reverse );
   bool triangleIntersection( const Primitive& triangle, const Ray& ray, float3& intersection, float3& normal, float3& areas, float& shadowIntensity,	bool& back );
   bool intersectionWithPrimitives( const Ray& ray, const int& iteration, int& closestPrimitive, float3& closestIntersection, float3& closestNormal, float3& closestAreas, float4& colorBox, bool& back, const int currentMaterialId);

protected:
   // Color management
   void   makeColor( float4& color, int index );
   float  processShadows( const float3& lampCenter, const float3& origin, const int& objectId, const int& iteration, float4& color);
   float4 intersectionShader( const Primitive& primitive, const float3& intersection, const float3& areas);
   float4 primitiveShader(const float3& origin, const float3& normal, const int& objectId, const float3& intersection, const float3& areas, const int& iteration, float4& refractionFromColor, float& shadowIntensity, float4& totalBlinn );

protected:
   // Rays
   float4 launchRay( const Ray& ray, float3& intersection, float& depthOfField, int4& primitiveXYId );

protected:
   // Post processing
   void k_standardRenderer();
   void k_fishEyeRenderer();
   void k_anaglyphRenderer();
   void k_3DVisionRenderer();
   void k_depthOfField();
   void k_ambiantOcclusion();
   void k_radiosity();
   void k_oneColor();
   void k_default();

private:
   float4* m_postProcessingBuffer;

};

