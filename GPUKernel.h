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
 * aint with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

/*
 * Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 */

#pragma once

#ifdef WIN32
   #include <Windows.h>
   #ifdef USE_KINECT
      #include <NuiApi.h>
   #endif // USE_KINECT
#endif // WIN32

#include <stdio.h>
#include <string>
#include <map>
#include <vector>

#include "Cuda/CudaDataTypes.h"
#include "DLL_API.h"

struct CPUPrimitive
{
   bool   movable;
	float3 p0;
	float3 p1;
	float3 p2;
	float3 n0;
	float3 n1;
	float3 n2;
	float3 size;
	int    type;
	int    materialId;
	float2 materialInfo;
   float3 speed;
};

struct CPUBoundingBox
{
   float3 parameters[2];
   std::vector<unsigned int> primitives;
};

typedef std::map<unsigned int,CPUBoundingBox> BoxContainer;
typedef std::map<unsigned int,CPUPrimitive>   PrimitiveContainer;
typedef std::map<unsigned int,unsigned int>   LampContainer;

class RAYTRACINGENGINE_API GPUKernel
{

public:
   GPUKernel( bool activeLogging, int platform = 0, int device = 0 );
   virtual ~GPUKernel();

   virtual void initBuffers();
   virtual void cleanup();

public:

   // ---------- Rendering ----------
	virtual void render_begin( const float timer ) = 0;
   virtual void render_end( char* bitmap) = 0;

public:

	// ---------- Primitives ----------
	int addPrimitive( PrimitiveType type );
	void setPrimitive( 
		unsigned int index,
		float x0, float y0, float z0, float w,  float h,  float d, int   materialId, float materialPaddingX, float materialPaddingY );
	void setPrimitive( 
		unsigned int index,
		float x0, float y0, float z0, float x1, float y1, float z1,
      float w,  float h,  float d, int   materialId, float materialPaddingX, float materialPaddingY );
	void setPrimitive( 
		unsigned int index,
		float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2, 
      float w,  float h,  float d, int   materialId, float materialPaddingX, float materialPaddingY );
   unsigned int getPrimitiveAt( int x, int y );
   void setPrimitiveIsMovable( unsigned int index, bool movable );

   void rotatePrimitives( float3 rotationCenter, float3 angles, unsigned int from, unsigned int to );
	void rotatePrimitive( CPUPrimitive& primitive, float3 rotationCenter, float3 cosAngles, float3 sinAngles );
   void rotateBox( CPUBoundingBox& box, float3 rotationCenter, float3 cosAngles, float3 sinAngles );

	void setPrimitiveMaterial( unsigned int index, int materialId); 
	int  getPrimitiveMaterial( unsigned int index); 
	float3 getPrimitiveCenter( unsigned int index );
	void getPrimitiveOtherCenter( unsigned int index, float3& otherCenter );
	void setPrimitiveCenter( unsigned int index, const float3& center );

   // Normals
	void setPrimitiveNormals( unsigned int index, float3 n0, float3 n1, float3 n2 );

   // Lights
   int getLight( unsigned int index );

   CPUPrimitive* getPrimitive( const unsigned int index );
     
public:
   
   void updateBoundingBox( CPUBoundingBox& box );
   void resetBoxes( bool resetPrimitives=true );
   void resetBox( CPUBoundingBox& box, bool resetPrimitives=true );
   CPUBoundingBox& getBoundingBox( const unsigned int boxIndex ) { return (*m_boundingBoxes)[boxIndex]; };
   int compactBoxes( bool reconstructBoxes=false );
   void displayBoxesInfo();

public:

	// ---------- Complex objects ----------
	int addCube( 
      unsigned int boxId,
		float x, float y, float z, 
		float radius, 
		int   materialId, 
		float materialPaddingX, float materialPaddingY );

	int addRectangle(
		unsigned int boxId,
		float x, float y, float z, 
		float w, float h, float d,
		int   materialId, 
		float materialPaddingX, float materialPaddingY );

public:

	// ---------- Materials ----------
	int addMaterial();
	void setMaterial( 
		unsigned int index,
		float r, float g, float b, float noise,
		float reflection, 
		float refraction,
		bool  procedural,
      bool  wireframe, int wireframeWidth,
		float transparency,
		int   textureId,
		float specValue, float specPower, float specCoef,
      float innerIllumination, float illuminationDiffusion, float illuminationPropagation, 
      bool  fastTransparency);

   int getMaterialAttributes( 
		unsigned int index,
		float& r, float& g, float& b, float& noise,
		float& reflection, 
		float& refraction,
		bool&  procedural,
      bool&  wireframe,
      int&   wireframeDepth,
		float& transparency,
		int&   textureId,
		float& specValue, float& specPower, float& specCoef,
      float& innerIllumination, float& illuminationDiffusion, float& illuminationPropagation );

   Material* getMaterial( const int index );

public:

	// ---------- Camera ----------
	void setCamera( 
		float3 eye, float3 dir, float3 angles );

public:

	// ---------- Textures ----------
	void setTexture(
		unsigned int index,
		char* texture );

	int addTexture( 
		const std::string& filename );

public:

   void setSceneInfo(
      int    width, int    height,
      float  transparentColor,
      bool   shadowsEnabled,
      float  viewDistance,
      float  shadowIntensity,
      int    nbRayIterations,
      float4 backgroundColor,
      bool   supportFor3DVision, float  width3DVision,
      bool   renderBoxes,
      int    pathTracingIteration, 
      int    maxPathTracingIterations,
      OutputType outputType,
      int    timer,
      int    fogEffect);

   // Scene
   void setSceneInfo( const SceneInfo& sceneInfo ) { m_sceneInfo = sceneInfo; };
   SceneInfo getSceneInfo() { return m_sceneInfo; };

   // Post processing
   void setPostProcessingInfo( 
      PostProcessingType type,
      float              param1,
      float              param2,
      int                param3 );
   void setPostProcessingInfo( const PostProcessingInfo& postProcessingInfo ) { m_postProcessingInfo = postProcessingInfo; }

public:

   // Vector Utilities
   float vectorLength( const float3& vector );
   void normalizeVector( float3& v );
   float3 crossProduct( const float3& b, const float3& c );
     
   // Bitmap export
   void saveBitmapToFile( const std::string& filename, char* bitmap, const int width, const int height, const int depth );


#ifdef USE_KINECT
public:
	// ---------- Kinect ----------

	int updateSkeletons( 
      unsigned int primitiveIndex,
		float3 skeletonPosition, 
		float size,
		float radius,       int materialId,
		float head_radius,  int head_materialId,
		float hands_radius, int hands_materialId,
		float feet_radius,  int feet_materialId);

	bool getSkeletonPosition( int index, float3& position );
   char* getDepthBitmap() { return m_hDepth; }
   char* getVideoBitmap() { return m_hVideo; }
#endif // USE_KINECT

public:

	unsigned int getNbActiveBoxes()      { return static_cast<unsigned int>(m_boundingBoxes->size()); };
	unsigned int getNbActivePrimitives() { return static_cast<unsigned int>(m_primitives->size()); };
	unsigned int getNbActiveLamps()      { return m_nbActiveLamps; };
	unsigned int getNbActiveMaterials()  { return m_nbActiveMaterials; };
	unsigned int getNbActiveTextures()   { return m_nbActiveTextures; };

public:

	char* loadFromFile( const std::string& filename, size_t& length );
   void  saveToFile( const std::string& filename, const std::string& content );

public:

   std::string getGPUDescription() { return m_gpuDescription; };

protected:
   
   unsigned int processBoxes( const unsigned int boxSize, unsigned int& nbActiveBoxes, bool simulate );

   void rotateVector( float3& v, const float3 rotationCenter, const float3& cosAngles, const float3& sinAngles );

protected:
 
   // GPU
	BoundingBox* m_hBoundingBoxes;
   Primitive*   m_hPrimitives;
	int*		    m_hLamps;
	Material*    m_hMaterials;
	char*        m_hTextures;
	float3*      m_hDepthOfField;
	float*	    m_hRandoms;
   int2*        m_hPrimitivesXYIds;

	unsigned int m_nbActivePrimitives;
	unsigned int m_nbActiveLamps;
	unsigned int m_nbActiveMaterials;
	unsigned int m_nbActiveTextures;
	float3		 m_viewPos;
	float3		 m_viewDir;
	float3		 m_angles;

protected:

	int	 m_draft;
   bool   m_primitivesTransfered;
	bool	 m_materialsTransfered;
	bool	 m_texturesTransfered;
   // Scene Size
   float3 m_minPos;
   float3 m_maxPos;

protected:

   // Scene
   SceneInfo m_sceneInfo;

   // Post Processing
   PostProcessingInfo m_postProcessingInfo;

protected:

   // activate or deactivate logging
   bool m_activeLogging; 

   // GPU information
   std::string m_gpuDescription;

protected:

   // CPU
	BoxContainer*       m_boundingBoxes;
	PrimitiveContainer* m_primitives;
	LampContainer*      m_lamps;

	// Kinect declarations
#ifdef USE_KINECT
protected:
   bool               m_kinectEnabled;
	HANDLE             m_skeletons;
	HANDLE             m_hNextDepthFrameEvent; 
	HANDLE             m_hNextVideoFrameEvent;
	HANDLE             m_hNextSkeletonEvent;
	HANDLE             m_pVideoStreamHandle;
	HANDLE             m_pDepthStreamHandle;
	NUI_SKELETON_FRAME m_skeletonFrame;

	int               m_skeletonIndex;
	int               m_skeletonsBody;
	int               m_skeletonsLamp;

   char* m_hVideo;
   char* m_hDepth;

#endif // USE_KINECT
};
