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
	float4 p0;
	float4 p1;
	float4 p2;
	float4 n0;
	float4 n1;
	float4 n2;
	float4 size;
	int    type;
	int    materialId;
	float2 materialInfo;
};

struct CPUBoundingBox
{
   float4 parameters[2];
   std::vector<int> primitives;
};

class RAYTRACINGENGINE_API GPUKernel
{

public:
   GPUKernel( bool activeLogging, int platform = 0, int device = 0 );
   virtual ~GPUKernel();

   virtual void initBuffers();

public:

   // ---------- Rendering ----------
	virtual void render_begin( const float timer ) = 0;
   virtual void render_end( char* bitmap) = 0;

public:

	// ---------- Primitives ----------
	int addPrimitive( PrimitiveType type );
	void setPrimitive( 
		int   index,
		float x0, float y0, float z0, float w,  float h,  float d, int   materialId, float materialPaddingX, float materialPaddingY );
	void setPrimitive( 
		int   index,
		float x0, float y0, float z0, float x1, float y1, float z1,
      float w,  float h,  float d, int   materialId, float materialPaddingX, float materialPaddingY );
	void setPrimitive( 
		int   index,
		float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2, 
      float w,  float h,  float d, int   materialId, float materialPaddingX, float materialPaddingY );
   int getPrimitiveAt( int x, int y );
   void setPrimitiveIsMovable( int index, bool movable );

   void rotatePrimitives( float4 rotationCenter, float4 angles, int from, int to );
	void rotatePrimitive( CPUPrimitive& primitive, float4 rotationCenter, float4 cosAngles, float4 sinAngles );
   void rotateBox( CPUBoundingBox& box, float4 rotationCenter, float4 cosAngles, float4 sinAngles );

	void setPrimitiveMaterial( int index, int materialId); 
	int  getPrimitiveMaterial( int index); 
	void getPrimitiveCenter( int index, float& x, float& y, float& z, float& w );
	void getPrimitiveOtherCenter( int index, float4& otherCenter );
	void setPrimitiveCenter( int index, float  x, float  y, float  z, float  w );

   // Normals
	void setPrimitiveNormals( int   index, float4 n0, float4 n1, float4 n2 );

   // Lights
   int getLight( int index );
	void setLight( 
		int   index,
		float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2, 
      float w,  float h,  float d, int   materialId, float materialPaddingX, float materialPaddingY );

   CPUPrimitive* getPrimitive( const int index );
     
public:
   
   void updateBoundingBox( CPUBoundingBox& box );
   void resetBoxes( bool resetPrimitives=true );
   void resetBox( CPUBoundingBox& box, bool resetPrimitives=true );
   CPUBoundingBox& getBoundingBox( const int boxIndex ) { return m_boundingBoxes[boxIndex]; };
   int compactBoxes( bool reconstructBoxes=false );
   void displayBoxesInfo();

public:

	// ---------- Complex objects ----------
	int addCube( 
      int boxId,
		float x, float y, float z, 
		float radius, 
		int   materialId, 
		float materialPaddingX, float materialPaddingY );

	int addRectangle(
		int boxId,
		float x, float y, float z, 
		float w, float h, float d,
		int   materialId, 
		float materialPaddingX, float materialPaddingY );

public:

	// ---------- Materials ----------
	int addMaterial();
	void setMaterial( 
		int   index,
		float r, float g, float b, float noise,
		float reflection, 
		float refraction,
		bool  procedural,
      bool  wireframe, int wireframeWidth,
		float transparency,
		int   textureId,
		float specValue, float specPower, float specCoef,
      float innerIllumination,
      bool  fastTransparency);

   int getMaterialAttributes( 
		int    index,
		float& r, float& g, float& b, float& noise,
		float& reflection, 
		float& refraction,
		bool&  procedural,
      bool&  wireframe,
      int&   wireframeDepth,
		float& transparency,
		int&   textureId,
		float& specValue, float& specPower, float& specCoef,
      float& innerIllumination );

   Material* getMaterial( const int index );

public:

	// ---------- Camera ----------
	void setCamera( 
		float4 eye, float4 dir, float4 angles );

public:

	// ---------- Textures ----------
	void setTexture(
		int   index,
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
   float vectorLength( const float4& vector );
   void normalizeVector( float4& v );
   float4 crossProduct( const float4& b, const float4& c );
     
   // Bitmap export
   void saveBitmapToFile( const std::string& filename, char* bitmap, const int width, const int height, const int depth );


#ifdef USE_KINECT
public:
	// ---------- Kinect ----------

	int updateSkeletons( 
      int    primitiveIndex,
		float4 skeletonPosition, 
		float size,
		float radius,       int materialId,
		float head_radius,  int head_materialId,
		float hands_radius, int hands_materialId,
		float feet_radius,  int feet_materialId);

	bool getSkeletonPosition( int index, float4& position );
   char* getDepthBitmap() { return m_hDepth; }
   char* getVideoBitmap() { return m_hVideo; }
#endif // USE_KINECT

public:

	int getNbActiveBoxes()      { return static_cast<int>(m_boundingBoxes.size()); };
	int getNbActivePrimitives() { return static_cast<int>(m_primitives.size()); };
	int getNbActiveints()       { return m_nbActiveLamps; };
	int getNbActiveMaterials()  { return m_nbActiveMaterials; };

public:

	char* loadFromFile( const std::string& filename, size_t& length );
   void  saveToFile( const std::string& filename, const std::string& content );

protected:
   
   int processBoxes( const int boxSize, int& nbActiveBoxes, bool simulate );

   void rotateVector( float4& v, const float4 rotationCenter, const float4& cosAngles, const float4& sinAngles );

protected:
 
   // GPU
	BoundingBox* m_hBoundingBoxes;
   Primitive*   m_hPrimitives;
	int*		    m_hLamps;
	Material*    m_hMaterials;
	char*        m_hTextures;
	float4*      m_hDepthOfField;
	float*	    m_hRandoms;
   int*         m_hPrimitivesXYIds;

	int			m_nbActivePrimitives;
	int			m_nbActiveLamps;
	int			m_nbActiveMaterials;
	int			m_nbActiveTextures;
	float4		m_viewPos;
	float4		m_viewDir;
	float4		m_angles;

protected:

	int	 m_draft;
	bool	 m_textureTransfered;
   // Scene Size
   float4 m_minPos;
   float4 m_maxPos;

protected:

   // Scene
   SceneInfo m_sceneInfo;

   // Post Processing
   PostProcessingInfo m_postProcessingInfo;

protected:

   bool m_activeLogging; // activate or deactivate logging

protected:

   // CPU
	std::map<int,CPUBoundingBox> m_boundingBoxes;
	std::map<int,CPUPrimitive> m_primitives;

	// Kinect declarations
#ifdef USE_KINECT
protected:
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
