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
   Vertex crossProduct( const Vertex& b, const Vertex& c );
   float vectorLength( const Vertex& v );
   float dot( const Vertex& v1, const Vertex& v2 );
   Vertex normalize( const Vertex& v );
   void vectorReflection( Vertex& r, const Vertex& i, const Vertex& n );
   void vectorRefraction( Vertex& refracted, const Vertex incident, const float n1, const Vertex normal, const float n2 );
   void vectorRotation( Vertex& v, const Vertex& rotationCenter, const Vertex& angles );

protected:
   void computeRayAttributes(Ray& ray);

   // Texture mapping
   void juliaSet( const Primitive& primitive, const float x, const float y, float4& color );
   void mandelbrotSet( const Primitive& primitive, const float x, const float y, float4& color );
   float4 sphereUVMapping( const Primitive& primitive, const Vertex& intersection);
   float4 triangleUVMapping( const Primitive& primitive, const Vertex& intersection, const Vertex& areas);
   float4 cubeMapping( const Primitive& primitive, const Vertex& intersection);
   bool wireFrameMapping( float x, float y, int width, const Primitive& primitive );

protected:
   // Intersections
   bool boxIntersection( const BoundingBox& box, const Ray& ray, const float& t0, const float& t1 );
   bool ellipsoidIntersection( const Primitive& ellipsoid, const Ray& ray, Vertex& intersection, Vertex& normal, float& shadowIntensity, bool& back);
   bool sphereIntersection( const Primitive& sphere, const Ray& ray, Vertex& intersection, Vertex& normal, float& shadowIntensity, bool& back );
   bool cylinderIntersection( const Primitive& cylinder, const Ray& ray, Vertex& intersection, Vertex& normal, float& shadowIntensity, bool& back );
   bool planeIntersection( const Primitive& primitive, const Ray& ray, Vertex& intersection, Vertex& normal, float& shadowIntensity, bool reverse );
   bool triangleIntersection( const Primitive& triangle, const Ray& ray, Vertex& intersection, Vertex& normal, Vertex& areas, float& shadowIntensity,	bool& back );
   bool intersectionWithPrimitives( const Ray& ray, const int& iteration, int& closestPrimitive, Vertex& closestIntersection, Vertex& closestNormal, Vertex& closestAreas, float4& colorBox, bool& back, const int currentMaterialId);

protected:
   // Color management
   void   makeColor( float4& color, int index );
   float  processShadows( const Vertex& lampCenter, const Vertex& origin, const int& objectId, const int& iteration, float4& color);
   float4 intersectionShader( const Primitive& primitive, const Vertex& intersection, const Vertex& areas);
   float4 primitiveShader(const Vertex& origin, const Vertex& normal, const int& objectId, const Vertex& intersection, const Vertex& areas, const int& iteration, float4& refractionFromColor, float& shadowIntensity, float4& totalBlinn, LightInformation* pathTracingInformation, const int pathTracingInformationSize, const bool isLightRay );

protected:
   // Rays
   float4 launchRay( const Ray& ray, Vertex& intersection, float& depthOfField, int4& primitiveXYId, LightInformation* pathTracingInformation, int& pathTracingInformationSize, const bool lightRay=false);

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

