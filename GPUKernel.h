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

#include "Consts.h"
#include "Cuda/CudaDataTypes.h"
#include "DLL_API.h"

#ifdef WIN32
   #ifdef USE_KINECT
      #include <NuiApi.h>
   #endif // USE_KINECT
#endif // WIN32

#ifdef USE_OCULUS
// Oculus Rift
#include <OVR.h>
#endif // USE_OCULUS

#include <map>
#include <vector>

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
   float3 vt0; // Texture coordinates
   float3 vt1; 
   float3 vt2; 
   float3 speed0;
   float3 speed1;
   float3 speed2;
};

struct CPUBoundingBox
{
   float3 parameters[2];
   std::vector<unsigned int> primitives;
};

typedef std::map<unsigned int,CPUBoundingBox> BoxContainer;
typedef std::map<unsigned int,CPUPrimitive>   PrimitiveContainer;
typedef std::map<unsigned int,Lamp>           LampContainer;
typedef std::vector<float3> Vertices;

class RAYTRACINGENGINE_API GPUKernel
{

public:
   GPUKernel( bool activeLogging, int optimalNbOfPrimmitivesPerBox, int platform, int device );
   virtual ~GPUKernel();

   virtual void initBuffers();
   virtual void cleanup();

public:

   // ---------- Rendering ----------
	virtual void render_begin( const float timer );
   virtual void render_end() = 0;
   BitmapBuffer* getBitmap() { return m_bitmap; };

public:

	// ---------- Primitives ----------
	int addPrimitive( PrimitiveType type );
	void setPrimitive( 
      const int& index,
		float x0, float y0, float z0, float w,  float h,  float d, int   materialId );
	void setPrimitive( 
      const int& index,
		float x0, float y0, float z0, float x1, float y1, float z1,
      float w,  float h,  float d, int   materialId );
	void setPrimitive( 
      const int& index,
		float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2, 
      float w,  float h,  float d, int   materialId );
   unsigned int getPrimitiveAt( 
      int x, int y );
   void setPrimitiveIsMovable( 
      const int& index,
      bool movable );

   // Scaling
   void scalePrimitives( float scale, unsigned int from, unsigned int to );

   // Rotation
   void rotatePrimitives( float3 rotationCenter, float3 angles, unsigned int from, unsigned int to );
	void rotatePrimitive( CPUPrimitive& primitive, float3 rotationCenter, float3 cosAngles, float3 sinAngles );
   void rotateBox( CPUBoundingBox& box, float3 rotationCenter, float3 cosAngles, float3 sinAngles );

   // Translation
   void translatePrimitives( float3 translation, unsigned int from, unsigned int to );

   // Morphing
   void morphPrimitives();

   // Material
	void setPrimitiveMaterial( unsigned int index, int materialId); 
	int  getPrimitiveMaterial( unsigned int index); 
	float3 getPrimitiveCenter( unsigned int index );
	void getPrimitiveOtherCenter( unsigned int index, float3& otherCenter );
	void setPrimitiveCenter( unsigned int index, const float3& center );

   // Texture coordinates
   void setPrimitiveTextureCoordinates( unsigned int index, float3 vt0, float3 vt1, float3 vt2 );

   // Normals
	void setPrimitiveNormals( unsigned int index, float3 n0, float3 n1, float3 n2 );

   // Lights
   int getLight( int index );

   CPUPrimitive* getPrimitive( const unsigned int index );
     
public:
   
   bool updateBoundingBox( CPUBoundingBox& box );
   void resetBoxes( bool resetPrimitives );
   void resetBox( CPUBoundingBox& box, bool resetPrimitives );
   CPUBoundingBox& getBoundingBox( const unsigned int boxIndex ) { return (*m_boundingBoxes)[m_frame][boxIndex]; };
   int compactBoxes( bool reconstructBoxes );
   void displayBoxesInfo(  );

public:

   // OpenGL
   int  setGLMode( const int& glMode, const int materialId );
   void addVertex( float x, float y, float z);
   void addNormal( float x, float y, float z);
   void addTextCoord( float x, float y, float z);
   void translate( float x, float y, float z);

public:

	// ---------- Complex objects ----------
	int addCube( 
		float x, float y, float z, 
		float radius, 
		int   materialId);

	int addRectangle(
		float x, float y, float z, 
		float w, float h, float d,
		int   materialId );

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

	void setMaterialColor(
		unsigned int index,
		float r, float g, float b );
	void setMaterialTextureId(
		unsigned int index, unsigned int textureId );

   int getMaterialAttributes( 
		int index,
		float& r, float& g, float& b, float& noise,
		float& reflection, 
		float& refraction,
		bool&  procedural,
      bool&  wireframe,
      int&   wireframeDepth,
		float& transparency,
		int&   textureId,
		float& specValue, float& specPower, float& specCoef,
      float& innerIllumination, float& illuminationDiffusion, float& illuminationPropagation,
      bool& fastTransparency);

   Material* getMaterial( const int index );

public:

	// ---------- Camera ----------
	void setCamera( 
		float3 eye, float3 dir, float3 angles );

public:

	// ---------- Textures ----------
	void setTexture( const int index, const TextureInformation& textureInfo );
   void setTexturesTransfered(const bool transfered) { m_texturesTransfered=transfered; };

	bool loadTextureFromFile( const int index, const std::string& filename );
   void buildLightInformationFromTexture( unsigned int index );

public:

   void setSceneInfo(
      int    width, int    height,
      float  transparentColor,
      int    shadowsEnabled,
      float  viewDistance,
      float  shadowIntensity,
      int    nbRayIterations,
      float4 backgroundColor,
      int    supportFor3DVision, float  width3DVision,
      bool   renderBoxes,
      int    pathTracingIteration, 
      int    maxPathTracingIterations,
      OutputType outputType,
      int    timer,
      int    fogEffect);

   // Scene
   void setSceneInfo( const SceneInfo& sceneInfo );
   SceneInfo getSceneInfo();

   // Post processing
   void setPostProcessingInfo( 
      PostProcessingType type,
      float              param1,
      float              param2,
      int                param3 );
   void setPostProcessingInfo( const PostProcessingInfo& postProcessingInfo );

public:

   // Vector Utilities
   float vectorLength( const float3& vector );
   void normalizeVector( float3& v );
   float3 crossProduct( const float3& b, const float3& c );
   float dotProduct( const float3& a, const float3& b );
     
   // Bitmap export
   void saveBitmapToFile( const std::string& filename, BitmapBuffer* bitmap, const int width, const int height, const int depth );

   // Oculus
protected:
#ifdef USE_OCULUS
   void initializeOVR();
   void finializeOVR();
private:
   // Oculus
   OVR::SensorFusion            m_sensorFusion;
   OVR::Ptr<OVR::SensorDevice>  m_sensor;
   OVR::Ptr<OVR::DeviceManager> m_manager;
   OVR::Ptr<OVR::HMDDevice>	  m_HMD;
#endif //  USE_OCULUS
   bool m_oculus; // True if Oculus is present and active


#ifdef USE_KINECT
public:
	// ---------- Kinect ----------
   void initializeKinectTextures();

	int updateSkeletons( 
      unsigned int primitiveIndex,
		float3 skeletonPosition, 
		float size,
		float radius,       int materialId,
		float head_radius,  int head_materialId,
		float hands_radius, int hands_materialId,
		float feet_radius,  int feet_materialId);

	bool getSkeletonPosition( int index, float3& position );
   BitmapBuffer* getDepthBitmap() { return m_hDepth; }
   BitmapBuffer* getVideoBitmap() { return m_hVideo; }
#endif // USE_KINECT

public:
   void setNbMaxPrimitivePerBox( const int nbMaxPrimitivePerBox ) { m_optimalNbOfPrimmitivesPerBox = nbMaxPrimitivePerBox; };
   unsigned int getNbActiveBoxes()      { return static_cast<unsigned int>(m_boundingBoxes[m_frame]->size()); };
   unsigned int getNbActivePrimitives() { return static_cast<unsigned int>(m_primitives[m_frame]->size()); };
	unsigned int getNbActiveLamps()      { return m_nbActiveLamps[m_frame]; };
	unsigned int getNbActiveMaterials()  { return m_nbActiveMaterials; };
	unsigned int getNbActiveTextures()   { return m_nbActiveTextures; };
   std::string  getTextureFilename( const int index ) { return m_textureFilenames[index]; };

   bool getProgressiveBoxes() { return m_progressiveBoxes; };
   void setProgressiveBoxes( const bool progressiveBoxes ) { m_progressiveBoxes = progressiveBoxes; };

   void resetAddingIndex() { m_addingIndex = 0; };
   void doneWithAdding( const bool& doneWithAdding ) {  m_doneWithAdding = doneWithAdding; };
   void resetFrame(  );
   void resetAll();

   void setDistortion( const float distortion ) { m_distortion = distortion; };

public:

	char* loadFromFile( const std::string& filename, size_t& length );
   void  saveToFile( const std::string& filename, const std::string& content );

public:

   std::string getGPUDescription() { return m_gpuDescription; };


public:
   void setNbFrames(const int nbFrames) { m_nbFrames=nbFrames; };
   int  getNbFrames() { return m_nbFrames; };
   void setFrame( const int frame ) { m_frame=frame; };
   int  getFrame() {return m_frame; };
   void nextFrame() 
   {
      m_frame++; 
      if(m_frame>=m_nbFrames) m_frame=m_nbFrames-1;
   };
   void previousFrame() 
   { 
      m_frame--; 
      if(m_frame<0) m_frame=0; 
   }

public:

   void rotateVector( float3& v, const float3& rotationCenter, const float3& cosAngles, const float3& sinAngles );

protected:
   
   int processBoxes( const int boxSize, int& nbActiveBoxes, bool simulate );

protected:
 
   // GPU
	BoundingBox* m_hBoundingBoxes;
   Primitive*   m_hPrimitives;
	int*		    m_hLamps;
	Material*    m_hMaterials;

   // Textures
   TextureInformation m_hTextures[NB_MAX_TEXTURES];
   std::map<int,std::string> m_textureFilenames;

   // Scene
	RandomBuffer*	       m_hRandoms;
   PrimitiveXYIdBuffer*  m_hPrimitivesXYIds;

   int m_nbActiveBoxes[NB_MAX_FRAMES];
	int m_nbActivePrimitives[NB_MAX_FRAMES];
	int m_nbActiveLamps[NB_MAX_FRAMES];
	int m_nbActiveMaterials;
   int m_nbActiveTextures;
   int m_lightInformationSize;
	float3		 m_viewPos;
	float3		 m_viewDir;
	float3		 m_angles;

   bool         m_doneWithAdding;
   int          m_addingIndex;

   // Distortion (Oculus)
   float m_distortion;

protected:
   int    m_frame;
   int    m_nbFrames;
   float  m_morph;

protected:

   // Rendering
   BitmapBuffer* m_bitmap;

protected:

	int	 m_draft;
   bool   m_primitivesTransfered;
	bool	 m_materialsTransfered;
	bool	 m_texturesTransfered;
   // Scene Size
   float3 m_minPos[NB_MAX_FRAMES];
   float3 m_maxPos[NB_MAX_FRAMES];

protected:

   // Scene
   SceneInfo m_sceneInfo;

   // Post Processing
   PostProcessingInfo m_postProcessingInfo;

   // Refresh
   bool m_refresh;

   // Progressive boxes
   // If true, only transfer visible boxes to the GPU
   bool m_progressiveBoxes;

protected:

   // activate or deactivate logging
   bool m_activeLogging; 

   // GPU information
   std::string m_gpuDescription;

   // GPUs & streams
   int2 m_occupancyParameters; 

protected:

   // CPU
	BoxContainer*       m_boundingBoxes[NB_MAX_FRAMES];
	PrimitiveContainer* m_primitives[NB_MAX_FRAMES];
	LampContainer*      m_lamps[NB_MAX_FRAMES];
   LightInformation*   m_lightInformation;

protected:
   int m_optimalNbOfPrimmitivesPerBox;

protected:
   // OpenGL
   int m_GLMode;
   Vertices m_vertices;
   Vertices m_normals;
   Vertices m_textCoords;
   float3   m_translation;

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

   BitmapBuffer* m_hVideo;
   BitmapBuffer* m_hDepth;

#endif // USE_KINECT
};
