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

#include "Cuda/CudaDataTypes.h"
#include "DLL_API.h"

class RAYTRACINGENGINE_API GPUKernel
{

public:
   GPUKernel( bool activeLogging, bool protein, int platform = 0, int device = 0 );
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
		int   index, int boxId,
		float x0, float y0, float z0, 
      float w,  float h,  float d,
		int   martialId, 
		int   materialPaddingX, int materialPaddingY );
	void setPrimitive( 
		int   index, int boxId,
		float x0, float y0, float z0, 
		float x1, float y1, float z1, 
      float w,  float h,  float d,
		int   martialId, 
		int   materialPaddingX, int materialPaddingY );
   int getPrimitiveAt( int x, int y );

   void rotatePrimitives( float4 rotationCenter, float4 angles, int from, int to );
	void rotatePrimitive( int boxId, int primitiveIndex, float4 rotationCenter, float4 cosAngles, float4 sinAngles );
	void translatePrimitive( int   index, int boxId, float x, float y, float z );
   void translatePrimitives( float x, float y, float z );
	void setPrimitiveMaterial( int index, int materialId); 
	int  getPrimitiveMaterial( int index); 
	void getPrimitiveCenter( int index, float& x, float& y, float& z, float& w );
	void setPrimitiveCenter( int index, int boxId, float  x, float  y, float  z, float  w );

public:
   
   void updateBoundingBox( const int boxId, const int primitiveIndex );
   void resetBox( int boxId, const bool resetPrimitives = true );
   BoundingBox& getBoundingBox( const int boxIndex ) { return m_hBoundingBoxes[boxIndex]; };
   int compactBoxes();
   void displayBoxesInfo();

public:

	// ---------- Complex objects ----------
	int addCube( 
      int boxId,
		float x, float y, float z, 
		float radius, 
		int   martialId, 
		int   materialPaddingX, int materialPaddingY );

	int addRectangle(
      int boxId,
		float x, float y, float z, 
      float w, float h, float d,
		int   martialId, 
		int   materialPaddingX, int materialPaddingY );

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
		float specValue, float specPower, 
      float innerIllumination, float specCoef );

   int getMaterial( 
		int    index,
		float& r, float& g, float& b, float& noise,
		float& reflection, 
		float& refraction,
		bool&  procedural,
      bool&  wireframe,
      int&   wireframeDepth,
		float& transparency,
		int&   textureId,
		float& specValue, float& specPower, 
      float& innerIllumination, float& specCoef );

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
      OutputType outputType);

   void setSceneInfo( const SceneInfo& sceneInfo ) { m_sceneInfo = sceneInfo; };

   void setPostProcessingInfo( 
      PostProcessingType type,
      float              param1,
      float              param2,
      int                param3 );
   void setPostProcessingInfo( const PostProcessingInfo& postProcessingInfo ) { m_postProcessingInfo = postProcessingInfo; }

   // lamps
   void resetLamps();

public:
   
   // Bitmap export
   void saveBitmapToFile( const std::string& filename, char* bitmap );


#ifdef USE_KINECT
public:
	// ---------- Kinect ----------

	int updateSkeletons( 
      int    primitiveIndex,
      int    boxId,
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

	int getNbActiveBoxes()      { return m_nbActiveBoxes; };
	int getNbActivePrimitives() { return m_nbActivePrimitives; };
	int getNbActiveints()       { return m_nbActiveLamps; };
	int getNbActiveMaterials()  { return m_nbActiveMaterials; };

public:

	char* loadFromFile( const std::string& filename, size_t& length );
   void  saveToFile( const std::string& filename, const std::string& content );

protected:
 
	BoundingBox* m_hBoundingBoxes;
   Primitive*   m_hPrimitives;
   int*         m_hBoxPrimitivesIndex;
	int*		    m_hLamps;
	Material*    m_hMaterials;
	char*        m_hTextures;
	float4*      m_hDepthOfField;
	float*	    m_hRandoms;
   int*         m_hPrimitivesXYIds;

   int         m_nbActiveBoxes;
	int			m_nbActivePrimitives;
	int			m_nbActiveLamps;
	int			m_nbActiveMaterials;
	int			m_nbActiveTextures;
	float4		m_viewPos;
	float4		m_viewDir;
	float4		m_angles;

protected:

	int			m_draft;
	bool		   m_textureTransfered;

protected:

   // Scene
   SceneInfo m_sceneInfo;

   // Post Processing
   PostProcessingInfo m_postProcessingInfo;

protected:

   bool m_activeLogging; // activate or deactivate logging
   bool m_protein; 


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
