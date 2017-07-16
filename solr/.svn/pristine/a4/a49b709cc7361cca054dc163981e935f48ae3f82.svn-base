/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Consts.h"
#include "Cuda/CudaDataTypes.h"
#include "DLL_API.h"

#ifdef WIN32
   #ifdef USE_KINECT
      #include <NuiApi.h>
   #endif // USE_KINECT
#else
#include <string>
#include <string.h>
#endif // WIN32

#ifdef USE_OCULUS
// Oculus Rift
#include <OVR.h>
#endif // USE_OCULUS

#include <map>
#include <vector>

struct CPUPrimitive
{
   bool   belongsToModel;
   bool   movable;
	Vertex p0;
	Vertex p1;
	Vertex p2;
	Vertex n0;
	Vertex n1;
	Vertex n2;
	Vertex size;
	int    type;
	int    materialId;
   Vertex vt0; // Texture coordinates
   Vertex vt1; 
   Vertex vt2; 
   Vertex speed0;
   Vertex speed1;
   Vertex speed2;
};

struct CPUBoundingBox
{
   Vertex parameters[2];
   Vertex center;
   std::vector<long> primitives;
   long indexForNextBox;
};

typedef std::map<long,CPUBoundingBox> BoxContainer;
typedef std::map<long,CPUPrimitive>   PrimitiveContainer;
typedef std::map<long,Lamp>           LampContainer;
typedef std::vector<Vertex> Vertices;

class RAYTRACINGENGINE_API GPUKernel
{

public:
   GPUKernel( bool activeLogging, int optimalNbOfBoxes );
   virtual ~GPUKernel();

   virtual void initBuffers();
   virtual void cleanup();
   
   virtual void reshape();

public:

   virtual void recompileKernels(const std::string& kernelCode="")=0;

public:

   // ---------- Rendering ----------
	virtual void render_begin( const float timer );
   virtual void render_end() = 0;
   BitmapBuffer* getBitmap() { return m_bitmap; };
   void generateScreenshot(const std::string& filename,const int width,const int height,const int quality);

public:

	// ---------- Primitives ----------
	int addPrimitive( PrimitiveType type, bool belongsToModel=false );
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
   void setPrimitiveBellongsToModel(
      const int& index,
      bool bellongsToModel );

   // Scaling
   void scalePrimitives( float scale, unsigned int from, unsigned int to );

   // Rotation
   void rotatePrimitives( const Vertex& rotationCenter, const Vertex& angles );
	void rotatePrimitive( CPUPrimitive& primitive, const Vertex& rotationCenter, const Vertex& cosAngles, const Vertex& sinAngles );
   void rotateBox( CPUBoundingBox& box, Vertex rotationCenter, Vertex cosAngles, Vertex sinAngles );
   Vertex getRotation() { return m_rotation; }

   // Translation
   void translatePrimitives( const Vertex& );
   Vertex getTranslation() { return m_translation; }

   // Morphing
   void morphPrimitives();

   // Material
	void setPrimitiveMaterial( unsigned int index, int materialId); 
	int  getPrimitiveMaterial( unsigned int index); 
	Vertex getPrimitiveCenter( unsigned int index );
	void getPrimitiveOtherCenter( unsigned int index, Vertex& otherCenter );
	void setPrimitiveCenter( unsigned int index, const Vertex& center );

   // Texture coordinates
   void setPrimitiveTextureCoordinates( unsigned int index, Vertex vt0, Vertex vt1, Vertex vt2 );

   // Normals
	void setPrimitiveNormals( unsigned int index, Vertex n0, Vertex n1, Vertex n2 );

   // Lights
   int getLight( int index );
   void reorganizeLights();
   CPUPrimitive* getPrimitive( const unsigned int index );
     
public:

   // OpenGL
   int  setGLMode( const int& glMode );
   void addVertex( float x, float y, float z);
   void addNormal( float x, float y, float z);
   void addTextCoord( float x, float y, float z);
   void translate( float x, float y, float z);
   void rotate( float x, float y, float z);

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
   void setMaterial(unsigned int index,const Material& material);
	void setMaterial( 
		unsigned int index,
		float r, float g, float b, float noise,
		float reflection, 
		float refraction,
		bool procedural,
      bool wireframe, int wireframeWidth,
		float transparency, float opacity,
	   int diffuseTextureId, int normalTextureId, int bumpTextureId, int specularTextureId, int reflectionTextureId, int transparentTextureId, int ambientOcclusionTextureId,
		float specValue, float specPower, float specCoef,
      float innerIllumination, float illuminationDiffusion, float illuminationPropagation, 
      bool fastTransparency);

	void setMaterialColor(
		unsigned int index,
		float r, float g, float b );
	void setMaterialTextureId(unsigned int textureId );

   int getMaterialAttributes( 
		int index,
		float& r, float& g, float& b, float& noise,
		float& reflection, 
		float& refraction,
		bool& procedural,
      bool& wireframe,
      int& wireframeDepth,
		float& transparency, float& opacity,
      int& diffuseTextureId, int& normalTextureId, int& bumpTextureId, int& specularTextureId, int& reflectionTextureId, int& transparencyTextureId, int& ambientOcclusionTextureId,
		float& specValue, float& specPower, float& specCoef,
      float& innerIllumination, float& illuminationDiffusion, float& illuminationPropagation,
      bool& fastTransparency);

   Material* getMaterial( const int index );
   int       getCurrentMaterial();
   void      setCurrentMaterial( const int currentMaterial );

public:

	// ---------- Camera ----------
	void setCamera( 
		Vertex eye, Vertex dir, Vertex angles );

public:

	// ---------- Textures ----------
	void setTexture( const int index, const TextureInformation& textureInfo );
	void getTexture( const int index, TextureInformation& textureInfo );
   void setTexturesTransfered(const bool transfered) { m_texturesTransfered=transfered; };
   void realignTexturesAndMaterials();

	bool loadTextureFromFile( const int index, const std::string& filename );
   void buildLightInformationFromTexture( unsigned int index );
   void processTextureOffsets();

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
      int    fogEffect,
      int    skyboxSize,
      int    skyboxMaterialId);

   // Scene
   void setSceneInfo( const SceneInfo& sceneInfo );
   SceneInfo& getSceneInfo();

   // Post processing
   void setPostProcessingInfo( 
      PostProcessingType type,
      float              param1,
      float              param2,
      int                param3 );
   void setPostProcessingInfo( const PostProcessingInfo& postProcessingInfo );

public:

   // Vector Utilities
   float vectorLength( const Vertex& vector );
   void normalizeVector( Vertex& v );
   Vertex crossProduct( const Vertex& b, const Vertex& c );
   float dotProduct( const Vertex& a, const Vertex& b );
     
   // Bitmap export
   void saveBitmapToFile( const std::string& filename, BitmapBuffer* bitmap, const int width, const int height, const int depth );

   // Oculus
public:
   void switchOculusVR();

protected:
#ifdef USE_OCULUS
   void initializeOVR();
   void finializeOVR();
private:
   // Oculus
   OVR::SensorFusion*           m_sensorFusion;
   OVR::Ptr<OVR::SensorDevice>  m_sensor;
   OVR::Ptr<OVR::DeviceManager> m_manager;
   OVR::Ptr<OVR::HMDDevice>     m_HMD;
#endif //  USE_OCULUS
   bool m_oculus; // True if Oculus is present and active


#ifdef USE_KINECT
public:
	// ---------- Kinect ----------
   void initializeKinectTextures();

	int updateSkeletons( 
      unsigned int primitiveIndex,
		Vertex skeletonPosition, 
		float size,
		float radius,       int materialId,
		float head_radius,  int head_materialId,
		float hands_radius, int hands_materialId,
		float feet_radius,  int feet_materialId);

	bool getSkeletonPosition( int index, Vertex& position );
   BitmapBuffer* getDepthBitmap() { return m_hDepth; }
   BitmapBuffer* getVideoBitmap() { return m_hVideo; }
#endif // USE_KINECT

public:
   void setOptimalNbOfBoxes( const int optimalNbOfBoxes );
   unsigned int getNbActiveBoxes();
   unsigned int getNbActivePrimitives();
	unsigned int getNbActiveLamps();
	unsigned int getNbActiveMaterials();
	unsigned int getNbActiveTextures();
   std::string  getTextureFilename( const int index );
   TextureInformation& getTextureInformation(const int index);


   void resetAddingIndex() { m_addingIndex = 0; };
   void doneWithAdding( const bool& doneWithAdding ) {  m_doneWithAdding = doneWithAdding; };
   void resetFrame();
   void resetAll();

   void setDistortion( const float distortion ) { m_distortion = distortion; };
   void setPointSize( const float pointSize );

public:

	//char* loadFromFile( const std::string& filename, size_t& length );
	void  loadFromFile(const std::string& filename);
   void  saveToFile(const std::string& filename);

public:

   virtual std::string getGPUDescription() = 0;


public:
   
   // Frames
   void setNbFrames(const int nbFrames);
   void setFrame( const int frame );
   int  getNbFrames();
   int  getFrame();
   void nextFrame();
   void previousFrame();

public:

   void rotateVector( Vertex& v, const Vertex& rotationCenter, const Vertex& cosAngles, const Vertex& sinAngles );

public:

   int  compactBoxes( bool reconstructBoxes );
   void streamDataToGPU(); 
   void displayBoxesInfo(  );
   void resetBoxes( bool resetPrimitives );

   void setTreeDepth( const int treeDepth );

protected:

   // Bounding boxes management
   int processBoxes( const int boxSize, bool simulate );
   int processOutterBoxes( const int boxSize, const int boundingBoxesDepth );
   bool updateBoundingBox( CPUBoundingBox& box );
   bool updateOutterBoundingBox( CPUBoundingBox& box, const int depth );
   void resetBox( CPUBoundingBox& box, bool resetPrimitives );

   void recursiveDataStreamToGPU( const int depth, std::vector<long>& elements );

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
   size_t m_maxPrimitivesPerBox;
	Vertex		 m_viewPos;
	Vertex		 m_viewDir;
	Vertex		 m_angles;

   bool         m_doneWithAdding;
   int          m_addingIndex;

   // Distortion (Oculus)
   float m_distortion;

protected:
   int    m_frame;
   int    m_nbFrames;
   float  m_morph;
   int    m_treeDepth;

protected:

   // Rendering
   BitmapBuffer* m_bitmap;

protected:

   bool   m_primitivesTransfered;
	bool	 m_materialsTransfered;
	bool	 m_texturesTransfered;
   bool   m_randomsTransfered;
   // Scene Size
   Vertex m_minPos[NB_MAX_FRAMES];
   Vertex m_maxPos[NB_MAX_FRAMES];

protected:

   // Scene
   SceneInfo m_sceneInfo;

   // Post Processing
   PostProcessingInfo m_postProcessingInfo;

   // Refresh
   bool m_refresh;

protected:

   // activate or deactivate logging
   bool m_activeLogging; 

   // GPU information
   std::string m_gpuDescription;

   // GPUs & streams
   int2 m_occupancyParameters; 

protected:

   // CPU
	BoxContainer        m_boundingBoxes[NB_MAX_FRAMES][BOUNDING_BOXES_TREE_DEPTH];
	PrimitiveContainer  m_primitives[NB_MAX_FRAMES];
	LampContainer       m_lamps[NB_MAX_FRAMES];
   LightInformation*   m_lightInformation;

protected:
   int m_optimalNbOfBoxes;

protected:
   // OpenGL
   int      m_GLMode;
   Vertices m_vertices;
   Vertices m_normals;
   Vertices m_textCoords;
   Vertex   m_translation;
   Vertex   m_rotation;
   int      m_currentMaterial;
   float    m_pointSize;

protected:

   // Benchmark
   long m_counter;

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
