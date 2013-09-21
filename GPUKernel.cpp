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

#ifdef WIN32
#include <windows.h>
#else
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#endif
#include <math.h>

#include <iostream>
#include <vector>

#include "OpenGL/rtgl.h"
#include "GPUKernel.h"
#include "Logging.h"
#include "Consts.h"
#include "ImageLoader.h"

const unsigned int MAX_SOURCE_SIZE = 65535*2;

float3 min4( const float3 a, const float3 b, const float3 c )
{
   float3 r;
   r.x = std::min(std::min(a.x,b.x),c.x);
   r.y = std::min(std::min(a.y,b.y),c.y);
   r.z = std::min(std::min(a.z,b.z),c.z);
   return r;
}

float3 max4( const float3 a, const float3 b, const float3 c )
{
   float3 r;
   r.x = std::max(std::max(a.x,b.x),c.x);
   r.y = std::max(std::max(a.y,b.y),c.y);
   r.z = std::max(std::max(a.z,b.z),c.z);
   return r;
}

 float GPUKernel::dotProduct( const float3& a, const float3& b )
 {
   return a.x * b.x + a.y * b.y + a.z * b.z;
 }

void vectorRotation( float3& v, const float3& rotationCenter, const float3& angles )
{ 
	float3 cosAngles, sinAngles;
	
   cosAngles.x = cos(angles.x);
	cosAngles.y = cos(angles.y);
	cosAngles.z = cos(angles.z);
	
   sinAngles.x = sin(angles.x);
	sinAngles.y = sin(angles.y);
	sinAngles.z = sin(angles.z);

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

// ________________________________________________________________________________
float GPUKernel::vectorLength( const float3& vector )
{
	return sqrt( vector.x*vector.x + vector.y*vector.y + vector.z*vector.z );
}

// ________________________________________________________________________________
void GPUKernel::normalizeVector( float3& v )
{
	float l = vectorLength( v );
   if( l != 0.f )
   {
      v.x /= l;
      v.y /= l;
      v.z /= l;
   }
}

float3 GPUKernel::crossProduct( const float3& b, const float3& c )
{
	float3 a;
	a.x = b.y*c.z - b.z*c.y;
	a.y = b.z*c.x - b.x*c.z;
	a.z = b.x*c.y - b.y*c.x;
	return a;
}

#ifndef WIN32
typedef struct {
	short bfType;
	int bfSize;
	short Reserved1;
	short Reserved2;
	int bfOffBits;
} BITMAPFILEHEADER;

typedef struct {
	int biSizeImage;
	int biWidth;
	int biHeight;
} BITMAPINFOHEADER;
#endif

GPUKernel::GPUKernel(bool activeLogging, int optimalNbOfPrimmitivesPerBox, int platform, int device)
 : m_optimalNbOfPrimmitivesPerBox(optimalNbOfPrimmitivesPerBox),
   m_hPrimitives(nullptr), 
	m_hMaterials(nullptr), 
   m_hLamps(nullptr),
   m_hBoundingBoxes(nullptr),
	m_hPrimitivesXYIds(nullptr),
	m_hRandoms(nullptr), 
	m_nbActiveMaterials(-1),
   m_nbActiveTextures(0),
   m_lightInformationSize(0),
	m_activeLogging(activeLogging),
   m_lightInformation(nullptr),
#if USE_KINECT
	m_hVideo(0), m_hDepth(0), 
	m_skeletons(0), m_hNextDepthFrameEvent(0), m_hNextVideoFrameEvent(0), m_hNextSkeletonEvent(0),
	m_pVideoStreamHandle(0), m_pDepthStreamHandle(0),
	m_skeletonsBody(-1), m_skeletonsLamp(-1), m_skeletonIndex(-1),
#endif // USE_KINECT
   m_primitivesTransfered(false),
	m_materialsTransfered(false),
   m_texturesTransfered(false),
   m_doneWithAdding(false),
   m_addingIndex(0),
   m_refresh(true),
   m_bitmap(nullptr),
   m_GLMode(-1),
   m_distortion(0.1f),
   m_frame(0)
{
   for( int i(0); i<NB_MAX_FRAMES; ++i)
   {
      m_primitives[i]=nullptr;
      m_boundingBoxes[i]=nullptr;
      m_lamps[i]=nullptr;
   	m_nbActiveBoxes[i]=0;
   	m_nbActivePrimitives[i]=0;
   	m_nbActiveLamps[i]=0;
   }

	LOG_INFO(3,"GPUKernel::GPUKernel (Log is " << (activeLogging ? "" : "de") << "activated" );
   LOG_INFO(1,"----------++++++++++  GPU Kernel created  ++++++++++----------" );

#if 1
	LOG_INFO(3, "CPU: SceneInfo         : " << sizeof(SceneInfo) );
	LOG_INFO(3, "CPU: Ray               : " << sizeof(Ray) );
	LOG_INFO(3, "CPU: PrimitiveType     : " << sizeof(PrimitiveType) );
	LOG_INFO(3, "CPU: Material          : " << sizeof(Material) );
	LOG_INFO(3, "CPU: BoundingBox       : " << sizeof(BoundingBox) );
	LOG_INFO(3, "CPU: Primitive         : " << sizeof(Primitive) );
	LOG_INFO(3, "CPU: PostProcessingType: " << sizeof(PostProcessingType) );
	LOG_INFO(3, "CPU: PostProcessingInfo: " << sizeof(PostProcessingInfo) );
	LOG_INFO(3, "Textures " << NB_MAX_TEXTURES );
#endif // 0

   m_progressiveBoxes = true;

#if USE_KINECT
	// Initialize Kinect
   LOG_INFO(1, "----------------------------" );
   LOG_INFO(1, "                         O  " );
   LOG_INFO(1, "                       --+--" );
   LOG_INFO(1, "                         |  " );
   LOG_INFO(1, "Kinect initialization   / \\" );
	HRESULT err=NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON | NUI_INITIALIZE_FLAG_USES_COLOR);
   m_kinectEnabled = (err==S_OK);

   if( m_kinectEnabled )
   {
	   m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	   m_hNextVideoFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL ); 
	   m_hNextSkeletonEvent   = CreateEvent( NULL, TRUE, FALSE, NULL );

	   m_skeletons = CreateEvent( NULL, TRUE, FALSE, NULL );			 
	   NuiSkeletonTrackingEnable( m_skeletons, NUI_SKELETON_TRACKING_FLAG_ENABLE_SEATED_SUPPORT );

	   NuiImageStreamOpen( NUI_IMAGE_TYPE_COLOR,                  NUI_IMAGE_RESOLUTION_640x480, 0, 2, m_hNextVideoFrameEvent, &m_pVideoStreamHandle );
	   NuiImageStreamOpen( NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_320x240, 0, 2, m_hNextDepthFrameEvent, &m_pDepthStreamHandle );

	   NuiCameraElevationSetAngle( 0 );
   }
   else
   {
      LOG_ERROR("    FAILED" );
   }
   LOG_INFO(1, "----------------------------" );
#endif // USE_KINECT
}


GPUKernel::~GPUKernel()
{
	LOG_INFO(3,"GPUKernel::~GPUKernel");

#if USE_KINECT
   if( m_kinectEnabled ) 
   {
      CloseHandle(m_skeletons);
      CloseHandle(m_hNextDepthFrameEvent); 
      CloseHandle(m_hNextVideoFrameEvent); 
      CloseHandle(m_hNextSkeletonEvent);
      NuiShutdown();
   }
#endif // USE_KINECT

   cleanup();
   LOG_INFO(1,"----------++++++++++ GPU Kernel Destroyed ++++++++++----------" );
   LOG_INFO(1,"" );
}

void GPUKernel::initBuffers()
{
	LOG_INFO(1,"GPUKernel::initBuffers");

   // Setup CPU resources
   for( int i(0); i<NB_MAX_FRAMES; ++i)
   {
      m_primitives[i]    = new PrimitiveContainer[NB_MAX_FRAMES];
      m_boundingBoxes[i] = new BoxContainer[NB_MAX_FRAMES];
      m_lamps[i]         = new LampContainer[NB_MAX_FRAMES];
   }

   // Textures
   for( int i(0); i<NB_MAX_TEXTURES; ++i )
   {
      memset(&m_hTextures[i],0,sizeof(TextureInformation));
   }

   m_lightInformation = new LightInformation[ NB_MAX_LIGHTINFORMATIONS ];

   // Primitive IDs
   int size = m_sceneInfo.width.x*m_sceneInfo.height.x;
   m_hPrimitivesXYIds = new PrimitiveXYIdBuffer[size];
   memset(m_hPrimitivesXYIds,0,size*sizeof(PrimitiveXYIdBuffer));

	m_hPrimitives = new Primitive[NB_MAX_PRIMITIVES];
	memset(m_hPrimitives,0,NB_MAX_PRIMITIVES*sizeof(Primitive) ); 
	m_hMaterials = new Material[NB_MAX_MATERIALS+1];
	memset(m_hMaterials,0,NB_MAX_MATERIALS*sizeof(Material) ); 
	m_hBoundingBoxes = new BoundingBox[NB_MAX_BOXES];
	memset(m_hBoundingBoxes,0,NB_MAX_BOXES*sizeof(BoundingBox));
	m_hLamps = new Lamp[NB_MAX_LAMPS];
	memset(m_hLamps,0,NB_MAX_LAMPS*sizeof(Lamp));

   // Randoms
	m_hRandoms = new RandomBuffer[size];
	int i;
#pragma omp parallel for
	for( i=0; i<size; ++i)
	{
		m_hRandoms[i] = (rand()%2000-1000)/80000.f;
	}

   // Bitmap
   m_bitmap = new BitmapBuffer[m_sceneInfo.width.x*m_sceneInfo.height.x*gColorDepth];
}

void GPUKernel::cleanup()
{
   LOG_INFO(1,"Cleaning up resources");

   for( int i(0); i<NB_MAX_FRAMES; ++i)
   {
      if( m_boundingBoxes[i] ) m_boundingBoxes[i]->clear();
      m_nbActiveBoxes[i] = 0;

      if( m_primitives[i] ) m_primitives[i]->clear();
      m_nbActivePrimitives[i] = 0;

      if(m_lamps[i]) m_lamps[i]->clear();
      m_nbActiveLamps[i] = 0;

      m_minPos[i].x =  100000.f;
      m_minPos[i].y =  100000.f;
      m_minPos[i].z =  100000.f;
      m_maxPos[i].x = -100000.f;
      m_maxPos[i].y = -100000.f;
      m_maxPos[i].z = -100000.f;
   }

   for( int i(0); i<NB_MAX_TEXTURES; ++i)
   {
#ifdef USE_KINECT
      if(i>1 && m_hTextures[i].buffer) delete [] m_hTextures[i].buffer;
#else
      if(m_hTextures[i].buffer) delete [] m_hTextures[i].buffer;
#endif // USE_KINECT
      m_hTextures[i].buffer = nullptr;
   }
   
   m_vertices.clear();
   m_normals.clear();
   m_textCoords.clear();

   if( m_hRandoms ) delete m_hRandoms; m_hRandoms = nullptr;
   if( m_bitmap ) delete [] m_bitmap; m_bitmap = nullptr;
   if(m_hBoundingBoxes) delete m_hBoundingBoxes; m_hBoundingBoxes = nullptr;
	if(m_hPrimitives) delete m_hPrimitives; m_hPrimitives = nullptr;
	if(m_hLamps) delete m_hLamps; m_hLamps = nullptr;
   if(m_hMaterials) delete m_hMaterials; m_hMaterials = nullptr;
	if(m_hPrimitivesXYIds) delete m_hPrimitivesXYIds; m_hPrimitivesXYIds = nullptr;
   if(m_lightInformation) delete m_lightInformation; m_lightInformation = nullptr;

	m_nbActiveMaterials = -1;
   m_nbActiveTextures = 0;
	m_materialsTransfered = false;
   m_primitivesTransfered = false;
   m_texturesTransfered = false;

   // Morphing
   m_morph = 0.f;
#if USE_KINECT
	m_hVideo = 0;
   m_hDepth = 0;
	m_skeletons = 0;
   m_hNextDepthFrameEvent = 0;
   m_hNextVideoFrameEvent = 0;
   m_hNextSkeletonEvent = 0;
	m_pVideoStreamHandle = 0;
   m_pDepthStreamHandle = 0;
	m_skeletonsBody = -1;
   m_skeletonsLamp = -1;
   m_skeletonIndex = -1;
#endif // USE_KINECT

}

/*
________________________________________________________________________________

Sets camera
________________________________________________________________________________
*/
void GPUKernel::setCamera( 
	float3 eye, float3 dir, float3 angles )
{
	LOG_INFO(3,"GPUKernel::setCamera(" << 
		eye.x << "," << eye.y << "," << eye.z << " -> " <<
		dir.x << "," << dir.y << "," << dir.z << " : "  <<
		angles.x << "," << angles.y << "," << angles.z << ")" 
		);
	m_viewPos   = eye;
	m_viewDir   = dir;
	m_angles.x  = angles.x;
	m_angles.y  = angles.y;
	m_angles.z  = angles.z;
   m_refresh   = true;
}

int GPUKernel::addPrimitive( PrimitiveType type )
{
	LOG_INFO(3,"GPUKernel::addPrimitive");
   int returnValue=-1;
   if( m_doneWithAdding )
   {
      returnValue = m_addingIndex;
      m_addingIndex++;
   }
   else
   {
      CPUPrimitive primitive;
      memset(&primitive,0,sizeof(CPUPrimitive));
      primitive.type = type;
      int index = static_cast<int>(m_primitives[m_frame]->size());
      (*m_primitives[m_frame])[index] = primitive;
      LOG_INFO(3,"m_primitives.size() = " << m_primitives[m_frame]->size());
	   returnValue = index;
   }
   return returnValue;
}

CPUPrimitive* GPUKernel::getPrimitive( const unsigned int index )
{
   CPUPrimitive* returnValue(NULL);
	if( index>=0 && index<=m_primitives[m_frame]->size()) 
   {
      returnValue = &(*m_primitives[m_frame])[index];
   }
   return returnValue;
}

void GPUKernel::setPrimitive( 
   const int& index,
	float x0, float y0, float z0, 
	float w,  float h,  float d,
	int   materialId )
{
	setPrimitive( index, x0, y0, z0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, materialId );
}

void GPUKernel::setPrimitive( 
   const int& index,
	float x0, float y0, float z0, 
	float x1, float y1, float z1, 
	float w,  float h,  float d,
	int   materialId )
{
	setPrimitive( index, x0, y0, z0, x1, y1, z1, 0.f, 0.f, 0.f, w, h, d, materialId );
}

void GPUKernel::setPrimitive( 
   const int& index,
	float x0, float y0, float z0, 
	float x1, float y1, float z1, 
	float x2, float y2, float z2, 
	float w,  float h,  float d,
	int   materialId )
{
   float scale = 1.f;
   float3 zero = {0.f,0.f,0.f};
   m_primitivesTransfered = false;
   /*
	LOG_INFO(3,"GPUKernel::setPrimitive( " << 
		index << " (" << 
		"center (" << x0 << "," << y0 << "," << z0 << ")," <<
		"size (" << w << "," << h << "," << d << ")," <<
		"material (" << materialId << "," << materialPaddingX << "," << materialPaddingY << ")"
		);
      */
   if( index>=0 && index<=m_primitives[m_frame]->size()) 
	{
      (*m_primitives[m_frame])[index].movable= true;
		(*m_primitives[m_frame])[index].p0.x   = x0*scale;
		(*m_primitives[m_frame])[index].p0.y   = y0*scale;
		(*m_primitives[m_frame])[index].p0.z   = z0*scale;
		(*m_primitives[m_frame])[index].p1.x   = x1*scale;
		(*m_primitives[m_frame])[index].p1.y   = y1*scale;
		(*m_primitives[m_frame])[index].p1.z   = z1*scale;
		(*m_primitives[m_frame])[index].p2.x   = x2*scale;
		(*m_primitives[m_frame])[index].p2.y   = y2*scale;
		(*m_primitives[m_frame])[index].p2.z   = z2*scale;
		(*m_primitives[m_frame])[index].size.x = w*scale;
		(*m_primitives[m_frame])[index].size.y = h*scale;
		(*m_primitives[m_frame])[index].size.z = d*scale;
      (*m_primitives[m_frame])[index].n0     = zero;
      (*m_primitives[m_frame])[index].n1     = zero;
      (*m_primitives[m_frame])[index].n2     = zero;
      (*m_primitives[m_frame])[index].vt0    = zero;
      (*m_primitives[m_frame])[index].vt1    = zero;
      (*m_primitives[m_frame])[index].vt2    = zero;
		(*m_primitives[m_frame])[index].materialId = materialId;

		switch( (*m_primitives[m_frame])[index].type )
		{
		case ptSphere:
         {
			   (*m_primitives[m_frame])[index].size.x = w*scale;
			   (*m_primitives[m_frame])[index].size.y = w*scale;
			   (*m_primitives[m_frame])[index].size.z = w*scale;
			   break;
         }
		case ptEllipsoid:
			{
		      (*m_primitives[m_frame])[index].size.x = w*scale;
		      (*m_primitives[m_frame])[index].size.y = h*scale;
		      (*m_primitives[m_frame])[index].size.z = d*scale;
            break;
         }
		case ptCylinder:
			{
				// Axis
				float3 axis;
				axis.x = x1*scale - x0*scale;
				axis.y = y1*scale - y0*scale;
				axis.z = z1*scale - z0*scale;
				float len = sqrt( axis.x*axis.x + axis.y*axis.y + axis.z*axis.z );
            if( len != 0.f )
            {
				   axis.x /= len;
				   axis.y /= len;
				   axis.z /= len;
            }
				(*m_primitives[m_frame])[index].n1.x = axis.x;
				(*m_primitives[m_frame])[index].n1.y = axis.y;
				(*m_primitives[m_frame])[index].n1.z = axis.z;

            (*m_primitives[m_frame])[index].size.x = w*scale; //(x1 - x0)/2.f;
            (*m_primitives[m_frame])[index].size.y = w*scale; //(y1 - y0)/2.f;
            (*m_primitives[m_frame])[index].size.z = w*scale; // (z1 - z0)/2.f;
				break;
			}
#ifdef USE_KINECT 
		case ptCamera:
			{
				(*m_primitives[m_frame])[index].n0.x = 0.f;
				(*m_primitives[m_frame])[index].n0.y = 0.f;
				(*m_primitives[m_frame])[index].n0.z = -1.f;
				break;
			}
#endif // USE_KINECT
		case ptXYPlane:
			{
				(*m_primitives[m_frame])[index].n0.x = 0.f;
				(*m_primitives[m_frame])[index].n0.y = 0.f;
				(*m_primitives[m_frame])[index].n0.z = 1.f;
				(*m_primitives[m_frame])[index].n1 = (*m_primitives[m_frame])[index].n0;
				(*m_primitives[m_frame])[index].n2 = (*m_primitives[m_frame])[index].n0;
				break;
			}
		case ptYZPlane:
			{
				(*m_primitives[m_frame])[index].n0.x = 1.f;
				(*m_primitives[m_frame])[index].n0.y = 0.f;
				(*m_primitives[m_frame])[index].n0.z = 0.f;
				(*m_primitives[m_frame])[index].n1 = (*m_primitives[m_frame])[index].n0;
				(*m_primitives[m_frame])[index].n2 = (*m_primitives[m_frame])[index].n0;
				break;
			}
		case ptXZPlane:
		case ptCheckboard:
			{
				(*m_primitives[m_frame])[index].n0.x = 0.f;
				(*m_primitives[m_frame])[index].n0.y = 1.f;
				(*m_primitives[m_frame])[index].n0.z = 0.f;
				(*m_primitives[m_frame])[index].n1 = (*m_primitives[m_frame])[index].n0;
				(*m_primitives[m_frame])[index].n2 = (*m_primitives[m_frame])[index].n0;
				break;
			}
		case ptTriangle:
			{
            float3 v0,v1;
            v0.x = (*m_primitives[m_frame])[index].p1.x-(*m_primitives[m_frame])[index].p0.x;
            v0.y = (*m_primitives[m_frame])[index].p1.y-(*m_primitives[m_frame])[index].p0.y;
            v0.z = (*m_primitives[m_frame])[index].p1.z-(*m_primitives[m_frame])[index].p0.z;
            normalizeVector(v0);
            
            v1.x = (*m_primitives[m_frame])[index].p2.x-(*m_primitives[m_frame])[index].p0.x;
            v1.y = (*m_primitives[m_frame])[index].p2.y-(*m_primitives[m_frame])[index].p0.y;
            v1.z = (*m_primitives[m_frame])[index].p2.z-(*m_primitives[m_frame])[index].p0.z;
            normalizeVector(v1);

            (*m_primitives[m_frame])[index].n0 = crossProduct(v0,v1);
            normalizeVector((*m_primitives[m_frame])[index].n0);
				(*m_primitives[m_frame])[index].n1 = (*m_primitives[m_frame])[index].n0;
				(*m_primitives[m_frame])[index].n2 = (*m_primitives[m_frame])[index].n0;
				break;
			}
		}
      //min
      m_minPos[m_frame].x = std::min(x0*scale,m_minPos[m_frame].x);
      m_minPos[m_frame].y = std::min(y0*scale,m_minPos[m_frame].y);
      m_minPos[m_frame].z = std::min(z0*scale,m_minPos[m_frame].z);
             
      // max
      m_maxPos[m_frame].x = std::max(x0*scale,m_maxPos[m_frame].x);
      m_maxPos[m_frame].y = std::max(y0*scale,m_maxPos[m_frame].y);
      m_maxPos[m_frame].z = std::max(z0*scale,m_maxPos[m_frame].z);
   }
	else
	{
		LOG_ERROR("GPUKernel::setPrimitive: Out of bounds (" << index << "/" << NB_MAX_PRIMITIVES << ")" );
	}
}

void GPUKernel::setPrimitiveIsMovable( const int& index, bool movable )
{
   if( index>=0 && index<m_primitives[m_frame]->size()) 
	{
      CPUPrimitive& primitive((*m_primitives[m_frame])[index]);
      primitive.movable = movable;
   }
}


void GPUKernel::setPrimitiveTextureCoordinates( unsigned int index, float3 vt0, float3 vt1, float3 vt2 )
{
   if( index>=0 && index<m_primitives[m_frame]->size()) 
	{
      CPUPrimitive& primitive((*m_primitives[m_frame])[index]);
      primitive.vt0 = vt0;
      primitive.vt1 = vt1;
      primitive.vt2 = vt2;

      /*
      std::cout << "(" 
         << vt0.x << "," << vt0.y << "," << vt0.z << " ; "
         << vt1.x << "," << vt1.y << "," << vt1.z << " ; "
         << vt2.x << "," << vt2.y << "," << vt2.z 
         << ")" << std::endl;
         */
   }
}


void GPUKernel::setPrimitiveNormals( int unsigned index, float3 n0, float3 n1, float3 n2 )
{
   if( index>=0 && index<m_primitives[m_frame]->size()) 
	{
      CPUPrimitive& primitive((*m_primitives[m_frame])[index]);
      normalizeVector(n0);
		primitive.n0 = n0;
      normalizeVector(n1);
		primitive.n1 = n1;
      normalizeVector(n2);
		primitive.n2 = n2;
   }
}

unsigned int GPUKernel::getPrimitiveAt( int x, int y )
{
	LOG_INFO(3,"GPUKernel::getPrimitiveAt(" << x << "," << y << ")" );
	unsigned int returnValue = -1;
	unsigned int index = y*m_sceneInfo.width.x+x;
	if( index>=0 && index<static_cast<unsigned int>(m_sceneInfo.width.x*m_sceneInfo.height.x))
   {
		returnValue = m_hPrimitivesXYIds[index].x;
   }
	return returnValue;
}

bool GPUKernel::updateBoundingBox( CPUBoundingBox& box )
{
	LOG_INFO(3,"GPUKernel::updateBoundingBox()" );

   bool result(false);

   // Process box size
	float3 corner0;
	float3 corner1;

   std::vector<unsigned int>::const_iterator it = box.primitives.begin();
   while( it != box.primitives.end() )
   {
      CPUPrimitive& primitive = (*m_primitives[m_frame])[*it];
      result = (m_hMaterials[primitive.materialId].innerIllumination.x != 0.f );
	   switch( primitive.type )
	   {
	   case ptTriangle: 
	   case ptCylinder: 
		   {
			   corner0 = min4(primitive.p0,primitive.p1,primitive.p2);
			   corner1 = max4(primitive.p0,primitive.p1,primitive.p2);
			   break;
		   }
	   default:
		   {
			   corner0 = primitive.p0;
			   corner1 = primitive.p0;
			   break;
		   }
	   }

	   float3 p0,p1;
	   p0.x = ( corner0.x <= corner1.x ) ? corner0.x : corner1.x;
	   p0.y = ( corner0.y <= corner1.y ) ? corner0.y : corner1.y;
	   p0.z = ( corner0.z <= corner1.z ) ? corner0.z : corner1.z;
	   p1.x = ( corner0.x >  corner1.x ) ? corner0.x : corner1.x;
	   p1.y = ( corner0.y >  corner1.y ) ? corner0.y : corner1.y;
	   p1.z = ( corner0.z >  corner1.z ) ? corner0.z : corner1.z;

	   switch( primitive.type )
	   {
	   case ptCylinder: 
	   case ptSphere: 
		   {
			   p0.x -= primitive.size.x;
			   p0.y -= primitive.size.x;
			   p0.z -= primitive.size.x;

			   p1.x += primitive.size.x;
			   p1.y += primitive.size.x;
			   p1.z += primitive.size.x;
			   break;
		   }
	   default:
		   {
			   p0.x -= primitive.size.x;
			   p0.y -= primitive.size.y;
			   p0.z -= primitive.size.z;
			   p1.x += primitive.size.x;
			   p1.y += primitive.size.y;
			   p1.z += primitive.size.z;
			   break;
		   }
	   }

	   if( p0.x < box.parameters[0].x ) box.parameters[0].x = p0.x;
	   if( p0.y < box.parameters[0].y ) box.parameters[0].y = p0.y;
	   if( p0.z < box.parameters[0].z ) box.parameters[0].z = p0.z;
	   if( p1.x > box.parameters[1].x ) box.parameters[1].x = p1.x;
	   if( p1.y > box.parameters[1].y ) box.parameters[1].y = p1.y;
	   if( p1.z > box.parameters[1].z ) box.parameters[1].z = p1.z;

      ++it;
   }
   return result;
}

void GPUKernel::resetBoxes( bool resetPrimitives )
{
   if( resetPrimitives )
   {
      for( int i(0); i<(m_boundingBoxes[m_frame])->size(); ++i )
      {
         resetBox( (*m_boundingBoxes[m_frame])[i], resetPrimitives );
      }
   }
   else
   {
      (m_boundingBoxes[m_frame])->clear();
   }
}

void GPUKernel::resetBox( CPUBoundingBox& box, bool resetPrimitives )
{
	LOG_INFO(3,"GPUKernel::resetBox(" << resetPrimitives << ")" );
	if( resetPrimitives ) 
	{
      box.primitives.clear();
	}

	box.parameters[0].x =  m_sceneInfo.viewDistance.x;
	box.parameters[0].y =  m_sceneInfo.viewDistance.x;
	box.parameters[0].z =  m_sceneInfo.viewDistance.x;
	box.parameters[1].x = -m_sceneInfo.viewDistance.x;
	box.parameters[1].y = -m_sceneInfo.viewDistance.x;
	box.parameters[1].z = -m_sceneInfo.viewDistance.x;
}

int GPUKernel::processBoxes( const int boxSize, int& nbActiveBoxes, bool simulate )
{
   float3 boxSteps;
   boxSteps.x = ( m_maxPos[m_frame].x - m_minPos[m_frame].x ) / boxSize;
   boxSteps.y = ( m_maxPos[m_frame].y - m_minPos[m_frame].y ) / boxSize;
   boxSteps.z = ( m_maxPos[m_frame].z - m_minPos[m_frame].z ) / boxSize;

   boxSteps.x = ( boxSteps.x == 0.f ) ? 1 : boxSteps.x;
   boxSteps.y = ( boxSteps.y == 0.f ) ? 1 : boxSteps.y;
   boxSteps.z = ( boxSteps.z == 0.f ) ? 1 : boxSteps.z;

   nbActiveBoxes = 0;
   std::map<unsigned int,unsigned int> primitivesPerBox;

   int optimalNbOfPrimmitivesPerBox = 5*static_cast<int>(sqrtf(static_cast<float>(m_primitives[m_frame]->size())));
   optimalNbOfPrimmitivesPerBox = (optimalNbOfPrimmitivesPerBox>480) ? 480 : optimalNbOfPrimmitivesPerBox;

   int p(0);
   std::map<unsigned int,CPUPrimitive>::iterator it = (m_primitives[m_frame])->begin();
   while( it != (m_primitives[m_frame])->end() )
   {
      CPUPrimitive& primitive((*it).second);

      float3 center=primitive.p0;
      /*
      center.x = (primitive.p0.x+primitive.p1.x+primitive.p2.x)/3.f;
      center.y = (primitive.p0.y+primitive.p1.y+primitive.p2.y)/3.f;
      center.z = (primitive.p0.z+primitive.p1.z+primitive.p2.z)/3.f;
      */

      int X = static_cast<int>(( center.x - m_minPos[m_frame].x ) / boxSteps.x);
      int Y = static_cast<int>(( center.y - m_minPos[m_frame].y ) / boxSteps.y);
      int Z = static_cast<int>(( center.z - m_minPos[m_frame].z ) / boxSteps.z);
      int B = X*boxSize*boxSize + Y*boxSize + Z;

      if( simulate )
      {

         if( primitivesPerBox.find(B) == primitivesPerBox.end() )
         {
            nbActiveBoxes++;
            primitivesPerBox[B] = 0;
         }
         else
         {
            primitivesPerBox[B]++;
         }
      }
      else
      {
         if( primitivesPerBox.find(B) == primitivesPerBox.end() )
         {
	         (*m_boundingBoxes[m_frame])[B].parameters[0].x = m_sceneInfo.viewDistance.x;
	         (*m_boundingBoxes[m_frame])[B].parameters[0].y = m_sceneInfo.viewDistance.x;
	         (*m_boundingBoxes[m_frame])[B].parameters[0].z = m_sceneInfo.viewDistance.x;
	         (*m_boundingBoxes[m_frame])[B].parameters[1].x = -m_sceneInfo.viewDistance.x;
	         (*m_boundingBoxes[m_frame])[B].parameters[1].y = -m_sceneInfo.viewDistance.x;
	         (*m_boundingBoxes[m_frame])[B].parameters[1].z = -m_sceneInfo.viewDistance.x;
         }
         (*m_boundingBoxes[m_frame])[B].primitives.push_back(p);
      }
      ++p;
      ++it;
   }

   int maxPrimitivePerBox(0);
   int delta = 0;
   if( simulate )
   {
      maxPrimitivePerBox = ( nbActiveBoxes != 0 ) ? static_cast<int>(m_primitives[m_frame]->size()/nbActiveBoxes) : 0;
      //delta = abs(optimalNbOfPrimmitivesPerBox-maxPrimitivePerBox);
      delta = abs(optimalNbOfPrimmitivesPerBox-nbActiveBoxes);
      LOG_INFO(2, "[" << boxSize << "/" << delta << "] Avg : " << maxPrimitivePerBox << " for " << nbActiveBoxes << " active boxes (" << m_primitives[m_frame]->size() << " primitives) - " << optimalNbOfPrimmitivesPerBox );
   }
   else
   {
      BoxContainer::iterator itb = (m_boundingBoxes[m_frame])->begin();
      while( itb != (m_boundingBoxes[m_frame])->end() )
      {
         updateBoundingBox((*itb).second);
         itb++;
      }
   }

   return delta;
}

int GPUKernel::compactBoxes( bool reconstructBoxes )
{
	LOG_INFO(3,"GPUKernel::compactBoxes (frame " << m_frame << ")" );
   m_primitivesTransfered = false;

   if( reconstructBoxes )
   {
      int activeBoxes(NB_MAX_BOXES);
#if 1
      LOG_INFO(1,"Constructing acceleration structures" );
      // Bounding boxes
      // Search for best trade-off
      std::map<unsigned int,unsigned int> primitivesPerBox;
      int maxPrimitivePerBox(0);
      int boxSize = 1024;
      int bestSize = boxSize;
      int bestActiveBoxes = 0;
      int bestRatio = 10000; 
      do 
      {
         int ratio = processBoxes(  boxSize, activeBoxes, true );
         if( ratio < bestRatio ) 
         {
            bestSize = boxSize;
            bestRatio = ratio;
            bestActiveBoxes = activeBoxes;
         }
         boxSize/=2;
      }
      while( boxSize>=2 );
      LOG_INFO(1, "Best trade off: " << bestSize << "/" << bestActiveBoxes << " boxes" );
      processBoxes(  bestSize, activeBoxes, false );
#else
      processBoxes(  64, activeBoxes, false );
#endif 0
   }

   // Transform data for ray-tracer
   float3 viewPos = m_viewPos;
   if( m_progressiveBoxes )
   {
      float3 rotationCenter = {0.f,0.f,0.f};
      float3 angles = m_angles;
      angles.x *= -1.f;
      vectorRotation( viewPos, rotationCenter, angles );
      m_primitivesTransfered = false;
   }

   m_nbActiveBoxes[m_frame] = 0;
   m_nbActivePrimitives[m_frame] = 0;
   m_nbActiveLamps[m_frame] = 0;
   if( m_boundingBoxes )
   {
      BoxContainer::iterator itb=(m_boundingBoxes[m_frame])->begin();
      while( itb != (m_boundingBoxes[m_frame])->end() )
      {
         CPUBoundingBox& box = (*itb).second;

         //bool addBox(true);
         if( m_progressiveBoxes )
         {
            // Only consider boxes that are not too far
            resetBox(box,false);
            bool containsLight = updateBoundingBox( box);

            float3 v1,v2;
            v1.x = box.parameters[0].x-viewPos.x;
            v1.y = box.parameters[0].y-viewPos.y;
            v1.z = box.parameters[0].z-viewPos.z;
            v2.x = box.parameters[1].x-viewPos.x;
            v2.y = box.parameters[1].y-viewPos.y;
            v2.z = box.parameters[1].z-viewPos.z;
            //float d = std::min(vectorLength(v1),vectorLength(v2));
            //addBox = (containsLight || (d<m_sceneInfo.viewDistance.x));
         }

         if( /*addBox && */(box.primitives.size()!=0) && (m_nbActiveBoxes[m_frame]<NB_MAX_BOXES) )
         {
            // Prepare boxes for GPU

            m_hBoundingBoxes[m_nbActiveBoxes[m_frame]].parameters[0] = box.parameters[0];
            m_hBoundingBoxes[m_nbActiveBoxes[m_frame]].parameters[1] = box.parameters[1];
            m_hBoundingBoxes[m_nbActiveBoxes[m_frame]].startIndex.x  = m_nbActivePrimitives[m_frame];
            m_hBoundingBoxes[m_nbActiveBoxes[m_frame]].nbPrimitives.x= static_cast<int>(box.primitives.size());

            std::vector<unsigned int>::const_iterator itp = box.primitives.begin();
            while( itp != box.primitives.end() )
            {
               // Prepare primitives for GPU
               if( (*itp) < NB_MAX_PRIMITIVES )
               {
                  CPUPrimitive& primitive = (*m_primitives[m_frame])[*itp];
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].index.x = (*itp);
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].type.x  = primitive.type;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].p0 = primitive.p0;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].p1 = primitive.p1;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].p2 = primitive.p2;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].n0 = primitive.n0;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].n1 = primitive.n1;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].n2 = primitive.n2;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].size = primitive.size;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].materialId.x = primitive.materialId;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].vt0 = primitive.vt0;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].vt1 = primitive.vt1;
                  m_hPrimitives[m_nbActivePrimitives[m_frame]].vt2 = primitive.vt2;

                  // Lights
                  if( primitive.materialId >= 0 && m_hMaterials[primitive.materialId].innerIllumination.x != 0.f )
                  {
                     m_hLamps[m_nbActiveLamps[m_frame]] = m_nbActivePrimitives[m_frame];
                     (*m_lamps[m_frame])[m_nbActiveLamps[m_frame]] = (*itp);
                     m_nbActiveLamps[m_frame]++;
                     LOG_INFO(3,"Lamp added (" << m_nbActiveLamps[m_frame] << "/" << NB_MAX_LAMPS << ")" );
                  }
               }
               ++itp;
               ++(m_nbActivePrimitives[m_frame]);
            }
            ++(m_nbActiveBoxes[m_frame]);
         }
         ++itb;
      }
      buildLightInformationFromTexture( 4);
      LOG_INFO( 3, "Compacted " << m_nbActiveBoxes[m_frame] << " boxes, and " << m_nbActivePrimitives[m_frame] << " primitives"); 
      if( m_nbActivePrimitives[m_frame] != m_primitives[m_frame]->size() )
      {
         LOG_ERROR("Lost primitives on the way... " << m_nbActivePrimitives[m_frame] << "!=" << m_primitives[m_frame]->size() );
      }
      return static_cast<int>(m_nbActiveBoxes[m_frame]);
   }
   return 0;
}

void GPUKernel::resetFrame()
{
   LOG_INFO(3, "Resetting frame " << m_frame);
   m_boundingBoxes[m_frame]->clear();
   m_nbActiveBoxes[m_frame] = 0;
   LOG_INFO(3, "Nb Boxes: " << m_boundingBoxes[m_frame]->size());

   m_primitives[m_frame]->clear();
   m_nbActivePrimitives[m_frame] = 0;
   LOG_INFO(3, "Nb Primitives: " << m_primitives[m_frame]->size());

   m_lamps[m_frame]->clear();
   m_nbActiveLamps[m_frame] = 0;
   LOG_INFO(3, "Nb Primitives: " << m_primitives[m_frame]->size());
}

void GPUKernel::resetAll()
{
	LOG_INFO(1,"Resetting frames" );

   int oldFrame(m_frame);
   for( int frame(0); frame<NB_MAX_FRAMES; ++frame)
   {
      m_frame = frame;
      resetFrame();
   }
   m_frame=oldFrame;
   m_primitivesTransfered = false;

	LOG_INFO(1,"Resetting textures and materials" );
   m_nbActiveMaterials = -1;
   m_materialsTransfered = false;

   for( int i(0); i<NB_MAX_TEXTURES; ++i )
   {
#ifdef USE_KINECT
      if( i>1 && m_hTextures[i].buffer ) delete [] m_hTextures[i].buffer;
#else
      if( m_hTextures[i].buffer ) delete [] m_hTextures[i].buffer;
#endif USE_KINECT
      memset(&m_hTextures[i],0,sizeof(TextureInformation));
   }
   m_nbActiveTextures = 0;
   m_texturesTransfered = false;
#ifdef USE_KINECT
   initializeKinectTextures();
#endif // USE_KINECT
}

void GPUKernel::displayBoxesInfo()
{
	for( unsigned int i(0); i<=m_boundingBoxes[m_frame]->size(); ++i )
	{
		CPUBoundingBox& box = (*m_boundingBoxes[m_frame])[i];
		LOG_INFO( 3, "Box " << i );
		LOG_INFO( 3, "- # of primitives: " << box.primitives.size() );
		LOG_INFO( 3, "- Corners 1      : " << box.parameters[0].x << "," << box.parameters[0].y << "," << box.parameters[0].z );
		LOG_INFO( 3, "- Corners 2      : " << box.parameters[1].x << "," << box.parameters[1].y << "," << box.parameters[1].z );
      unsigned int p(0);
      std::vector<unsigned int>::const_iterator it = box.primitives.begin();
      while( it != box.primitives.end() )
      {
         CPUPrimitive& primitive((*m_primitives[m_frame])[*it]);
			LOG_INFO( 3, "- - " << 
				p << ":" << 
				"type = " << primitive.type << ", " << 
				"center = (" << primitive.p0.x << "," << primitive.p0.y << "," << primitive.p0.z << "), " << 
				"p1 = (" << primitive.p1.x << "," << primitive.p1.y << "," << primitive.p1.z << ")" );
         ++p;
         ++it;
		}
	}
}

void GPUKernel::rotatePrimitives( float3 rotationCenter, float3 angles, unsigned int from, unsigned int to )
{
	LOG_INFO(3,"GPUKernel::rotatePrimitives(" << from << "->" << to << ")" );
   m_primitivesTransfered = false;
	float3 cosAngles, sinAngles;
	
   cosAngles.x = cos(angles.x);
	cosAngles.y = cos(angles.y);
	cosAngles.z = cos(angles.z);
	
   sinAngles.x = sin(angles.x);
	sinAngles.y = sin(angles.y);
	sinAngles.z = sin(angles.z);

#pragma omp parallel
   for( BoxContainer::iterator itb=m_boundingBoxes[m_frame]->begin(); itb!=m_boundingBoxes[m_frame]->end(); ++itb )
   {
      #pragma omp single nowait
      {
         CPUBoundingBox& box = (*itb).second;
         resetBox(box,false);

         for( std::vector<unsigned int>::iterator it=box.primitives.begin(); it!=box.primitives.end(); ++it )
		   {
            //#pragma single nowait
            CPUPrimitive& primitive((*m_primitives[m_frame])[*it]);
            if( primitive.movable && primitive.type != ptCamera )
            {
#if 0
               float limit = -3000.f;
               if( primitive.speed0.y != 0.f && (primitive.p0.y > limit || primitive.p1.y > limit || primitive.p2.y > limit) )
               {
                  // Fall
                  primitive.p0.y += primitive.speed0.y;
                  primitive.p1.y += primitive.speed0.y;
                  primitive.p2.y += primitive.speed0.y;

                  if( primitive.p0.y < limit-primitive.size.x ) 
                  {
                     primitive.speed0.y = -primitive.speed0.y/1.6f;
                     // Rotate
                     float3 center = (primitive.p0.y < primitive.p1.y) ? primitive.p0 : primitive.p1;
                     center = (primitive.p2.y < center.y) ? primitive.p2 : center;


                     //center.x = (primitive.p0.x + primitive.p1.x + primitive.p2.x) / 3.f;
                     // center.y = (primitive.p0.y + primitive.p1.y + primitive.p2.y) / 3.f;
                     //center.z = (primitive.p0.z + primitive.p1.z + primitive.p2.z) / 3.f;

                     rotatePrimitive( primitive, center, cosAngles, sinAngles );
                  }
                  primitive.speed0.y -= 4+(rand()%400/100.f);
               }
               else
               {
                  primitive.speed0.y = 0.f;
               }
#else
               rotatePrimitive( primitive, rotationCenter, cosAngles, sinAngles );
#endif // 0
            }
         }
      }
   }
}

void GPUKernel::translatePrimitives( float3 translation, unsigned int from, unsigned int to )
{
	LOG_INFO(3,"GPUKernel::translatePrimitives(" << from << "->" << to << ")" );
   m_primitivesTransfered = false;
#pragma omp parallel
   for( BoxContainer::iterator itb=m_boundingBoxes[m_frame]->begin(); itb!=m_boundingBoxes[m_frame]->end(); ++itb )
   {
      #pragma omp single nowait
      {
         CPUBoundingBox& box = (*itb).second;
         for( int i(0); i<2; ++i )
         {
            box.parameters[i].x += translation.x;
            box.parameters[i].y += translation.y;
            box.parameters[i].z += translation.z;
         }

         for( std::vector<unsigned int>::iterator it=box.primitives.begin(); it!=box.primitives.end(); ++it )
		   {
            //#pragma single nowait
            CPUPrimitive& primitive((*m_primitives[m_frame])[*it]);
            if( primitive.movable && primitive.type != ptCamera )
            {
               primitive.p0.x += translation.x;
               primitive.p0.y += translation.y;
               primitive.p0.z += translation.z;

               primitive.p1.x += translation.x;
               primitive.p1.y += translation.y;
               primitive.p1.z += translation.z;

               primitive.p2.x += translation.x;
               primitive.p2.y += translation.y;
               primitive.p2.z += translation.z;
            }
         }
      }
   }
}

void GPUKernel::morphPrimitives()
{
//#pragma omp parallel
   for( int frame(1); frame<(m_nbFrames-1); ++frame )
   {
      //#pragma omp single nowait
      {
         setFrame(frame);
         PrimitiveContainer::iterator it2=m_primitives[m_nbFrames-1]->begin();
         for( PrimitiveContainer::iterator it1=m_primitives[0]->begin(); it1!=m_primitives[0]->end() && it2!=m_primitives[m_nbFrames-1]->end(); ++it1 )
         {
            CPUPrimitive& primitive1((*it1).second);
            CPUPrimitive& primitive2((*it2).second);
            float3 p0,p1,p2;
            float3 n0,n1,n2;
            float3 size;
            float r = static_cast<float>(m_frame)/static_cast<float>(m_nbFrames);
            p0.x = primitive1.p0.x+r*(primitive2.p0.x - primitive1.p0.x);
            p0.y = primitive1.p0.y+r*(primitive2.p0.y - primitive1.p0.y);
            p0.z = primitive1.p0.z+r*(primitive2.p0.z - primitive1.p0.z);

            p1.x = primitive1.p1.x+r*(primitive2.p1.x - primitive1.p1.x);
            p1.y = primitive1.p1.y+r*(primitive2.p1.y - primitive1.p1.y);
            p1.z = primitive1.p1.z+r*(primitive2.p1.z - primitive1.p1.z);

            p2.x = primitive1.p2.x+r*(primitive2.p2.x - primitive1.p2.x);
            p2.y = primitive1.p2.y+r*(primitive2.p2.y - primitive1.p2.y);
            p2.z = primitive1.p2.z+r*(primitive2.p2.z - primitive1.p2.z);

            n0.x = primitive1.n0.x+r*(primitive2.n0.x - primitive1.n0.x);
            n0.y = primitive1.n0.y+r*(primitive2.n0.y - primitive1.n0.y);
            n0.z = primitive1.n0.z+r*(primitive2.n0.z - primitive1.n0.z);

            n1.x = primitive1.n1.x+r*(primitive2.n1.x - primitive1.n1.x);
            n1.y = primitive1.n1.y+r*(primitive2.n1.y - primitive1.n1.y);
            n1.z = primitive1.n1.z+r*(primitive2.n1.z - primitive1.n1.z);

            n2.x = primitive1.n2.x+r*(primitive2.n2.x - primitive1.n2.x);
            n2.y = primitive1.n2.y+r*(primitive2.n2.y - primitive1.n2.y);
            n2.z = primitive1.n2.z+r*(primitive2.n2.z - primitive1.n2.z);

            size.x = primitive1.size.x+r*(primitive2.size.x - primitive1.size.x);
            size.y = primitive1.size.y+r*(primitive2.size.y - primitive1.size.y);
            size.z = primitive1.size.z+r*(primitive2.size.z - primitive1.size.z);

            int i = addPrimitive( PrimitiveType(primitive1.type) );
            setPrimitive(i, 
               p0.x, p0.y, p0.z,
               p1.x, p1.y, p1.z,
               p2.x, p2.y, p2.z,
               size.x,size.y,size.z,
               primitive1.materialId );
            
            setPrimitiveNormals(i, n0,n1,n2);
            setPrimitiveTextureCoordinates(i, primitive1.vt0,primitive1.vt1,primitive1.vt2);

            setPrimitiveIsMovable( i, primitive1.movable );

            ++it2;
         }
         compactBoxes(true);
      }
   }
}

void GPUKernel::scalePrimitives( float scale, unsigned int from, unsigned int to )
{
	LOG_INFO(3,"GPUKernel::rotatePrimitives(" << from << "->" << to << ")" );
   m_primitivesTransfered = false;

   PrimitiveContainer::iterator it = (*m_primitives[m_frame]).begin();
   while( it != (*m_primitives[m_frame]).end() )
	{
      CPUPrimitive& primitive((*it).second);
		primitive.p0.x *= scale;
		primitive.p0.y *= scale;
		primitive.p0.z *= scale;

      primitive.p1.x *= scale;
		primitive.p1.y *= scale;
		primitive.p1.z *= scale;

      primitive.p2.x *= scale;
		primitive.p2.y *= scale;
		primitive.p2.z *= scale;

      primitive.size.x *= scale;
		primitive.size.y *= scale;
		primitive.size.z *= scale;
      ++it;
	}
}

void GPUKernel::rotateVector( float3& v, const float3& rotationCenter, const float3& cosAngles, const float3& sinAngles )
{
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

void GPUKernel::rotatePrimitive( 
	CPUPrimitive& primitive, float3 rotationCenter, float3 cosAngles, float3 sinAngles )
{
	LOG_INFO(3,"GPUKernel::rotatePrimitive" );
   if( primitive.type == ptTriangle || primitive.type == ptSphere || primitive.type == ptEllipsoid || primitive.type == ptCylinder )
	{
      rotateVector( primitive.p0, rotationCenter, cosAngles, sinAngles );
		if( primitive.type == ptCylinder || primitive.type == ptTriangle )
		{
         rotateVector( primitive.p1, rotationCenter, cosAngles, sinAngles );
         rotateVector( primitive.p2, rotationCenter, cosAngles, sinAngles );
         // Rotate Normals
         float3 zeroCenter = {0.f,0.f,0.f};
         rotateVector( primitive.n0, zeroCenter, cosAngles, sinAngles );
         rotateVector( primitive.n1, zeroCenter, cosAngles, sinAngles );
         rotateVector( primitive.n2, zeroCenter, cosAngles, sinAngles );
         if( primitive.type == ptCylinder )
         {
            // Axis
				float3 axis;
				axis.x = primitive.p1.x - primitive.p0.x;
				axis.y = primitive.p1.y - primitive.p0.y;
				axis.z = primitive.p1.z - primitive.p0.z;
				float len = sqrtf( axis.x*axis.x + axis.y*axis.y + axis.z*axis.z );
            if( len != 0 )
            {
				   axis.x /= len;
				   axis.y /= len;
				   axis.z /= len;
            }
				primitive.n1.x = axis.x;
				primitive.n1.y = axis.y;
				primitive.n1.z = axis.z;
         }
		}
	}
}

void GPUKernel::rotateBox( 
	CPUBoundingBox& box, float3 rotationCenter, float3 cosAngles, float3 sinAngles )
{
	LOG_INFO(3,"GPUKernel::rotateBox" );
   rotateVector( box.parameters[0], rotationCenter, cosAngles, sinAngles );
   rotateVector( box.parameters[1], rotationCenter, cosAngles, sinAngles );
}

float3 GPUKernel::getPrimitiveCenter(
   unsigned int index)
{
   float3 center = {0.f,0.f,0.f};
	if( index>=0 && index<=m_primitives[m_frame]->size()) 
	{
		center.x = (*m_primitives[m_frame])[index].p0.x;
		center.y = (*m_primitives[m_frame])[index].p0.y;
		center.z = (*m_primitives[m_frame])[index].p0.z;
	}
   return center;
}

void GPUKernel::getPrimitiveOtherCenter(
	unsigned int index, float3& center)
{
	if( index>=0 && index<=m_primitives[m_frame]->size()) 
	{
		center = (*m_primitives[m_frame])[index].p1;
	}
}

void GPUKernel::setPrimitiveCenter(
	unsigned int index, const float3& center)
{
   m_primitivesTransfered = false;

   // TODO, Box needs to be updated
	if( index>=0 && index<=m_primitives[m_frame]->size()) 
	{
		(*m_primitives[m_frame])[index].p0.x = center.x;
		(*m_primitives[m_frame])[index].p0.y = center.y;
		(*m_primitives[m_frame])[index].p0.z = center.z;
	}
}

int GPUKernel::addCube( 
	float x, float y, float z, 
	float radius, 
	int   materialId )
{
	LOG_INFO(3,"GPUKernel::addCube(" << m_frame << ")" );
	return addRectangle( x,y,z,radius,radius,radius,materialId);
}

int GPUKernel::addRectangle( 
	float x, float y, float z, 
	float w, float h, float d,
	int   materialId )
{
	LOG_INFO(3,"GPUKernel::addRectangle(" << m_frame << ")" );
	int returnValue;
	// Back
	returnValue = addPrimitive(  ptXYPlane );
	setPrimitive(  returnValue, x, y, z+d, w, h, d, materialId ); 

	// Front
	returnValue = addPrimitive(  ptXYPlane );
	setPrimitive(  returnValue, x, y, z-d, w, h, d, materialId ); 

	// Left
	returnValue = addPrimitive(  ptYZPlane );
	setPrimitive(  returnValue, x-w, y, z, w, h, d, materialId ); 

	// Right
	returnValue = addPrimitive(  ptYZPlane );
	setPrimitive(  returnValue, x+w, y, z, w, h, d, materialId ); 

	// Top
	returnValue = addPrimitive(  ptXZPlane );
	setPrimitive(  returnValue, x, y+h, z, w, h, d, materialId ); 

	// Bottom
	returnValue = addPrimitive(  ptXZPlane );
	setPrimitive(  returnValue, x, y-h, z, w, h, d, materialId ); 
	return returnValue;
}

void GPUKernel::setPrimitiveMaterial( 
	unsigned int index, 
	int   materialId)
{
	LOG_INFO(3,"GPUKernel::setPrimitiveMaterial(" << index << "," << materialId << ")" );
	if( index>=0 && index<=m_primitives[m_frame]->size()) 
	{
		(*m_primitives[m_frame])[index].materialId = materialId;
		// TODO: updateLight( index );
	}
}

int GPUKernel::getPrimitiveMaterial( unsigned int index )
{
	LOG_INFO(3,"GPUKernel::getPrimitiveMaterial(" << index << ")" );
	unsigned int returnValue(-1);
	if( index>=0 && index<=m_primitives[m_frame]->size()) 
	{
		returnValue = (*m_primitives[m_frame])[index].materialId;
	}
	return returnValue;
}

// ---------- Materials ----------
int GPUKernel::addMaterial()
{
	LOG_INFO(3,"GPUKernel::addMaterial" );
	m_nbActiveMaterials++;
	LOG_INFO(3,"m_nbActiveMaterials = " << m_nbActiveMaterials );
	return m_nbActiveMaterials;
}

void GPUKernel::setMaterial( 
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
   bool fastTransparency)
{
	LOG_INFO(3,"GPUKernel::setMaterial( " << 
		index << "," <<
		"color=(" << r << "," << g << "," << b << ")," <<
		"reflection=" << reflection << "," <<
		"refraction=" << refraction << "," <<
		"transparency=" << transparency << "," <<
		"procedural=" << (procedural ? "true" : "false") << "," <<
		"wireframe=" << (wireframe ? "true" : "false") << "," <<
		"textureId=" << textureId << "," <<
		"specular=(" << specValue << "," << specPower << "," << specCoef << ")," << 
		"innerIllumination=" << innerIllumination << "," 
		"fastTransparency=" << fastTransparency
		);

   if( index>=0 && index<NB_MAX_MATERIALS ) 
	{
		m_hMaterials[index].color.x     = r;
		m_hMaterials[index].color.y     = g;
		m_hMaterials[index].color.z     = b;
		m_hMaterials[index].color.w     = noise;
		m_hMaterials[index].specular.x  = specValue;
		m_hMaterials[index].specular.y  = specPower;
		m_hMaterials[index].specular.z  = 0.f; // Not used
		m_hMaterials[index].specular.w  = specCoef;
		m_hMaterials[index].innerIllumination.x = innerIllumination;
		m_hMaterials[index].innerIllumination.y = illuminationDiffusion;
		m_hMaterials[index].innerIllumination.z = illuminationPropagation;
		m_hMaterials[index].reflection.x  = reflection;
		m_hMaterials[index].refraction.x  = refraction;
		m_hMaterials[index].transparency.x= transparency;
      m_hMaterials[index].attributes.x = fastTransparency ? 1 : 0;
		m_hMaterials[index].attributes.y = procedural ? 1 : 0;
		m_hMaterials[index].attributes.z = wireframe ? ((wireframeWidth == 0 ) ? 1 : 2) : 0;
		m_hMaterials[index].attributes.w = wireframeWidth;
		m_hMaterials[index].textureMapping.x = 1;
		m_hMaterials[index].textureMapping.y = 1;
		m_hMaterials[index].textureMapping.z = textureId;
      if( textureId>=0 && textureId<NB_MAX_TEXTURES )
      {
         m_hMaterials[index].textureMapping.x = m_hTextures[textureId].size.x; // Width
         m_hMaterials[index].textureMapping.y = m_hTextures[textureId].size.y; // Height
         m_hMaterials[index].textureMapping.w = m_hTextures[textureId].size.z; // Depth
         m_hMaterials[index].textureOffset.x  = m_hTextures[textureId].offset; // Offset
      }
      else
      {
         // Computed textures (Mandelbrot, Julia, etc)
         m_hMaterials[index].textureMapping.x = 40000;
         m_hMaterials[index].textureMapping.y = 40000;
         m_hMaterials[index].textureMapping.w = 3;
         m_hMaterials[index].textureOffset.x  = 0;
      }

		m_materialsTransfered = false;
	}
	else
	{
		LOG_ERROR("GPUKernel::setMaterial: Out of bounds(" << index << "/" << NB_MAX_MATERIALS << ")" );
	}
}

int GPUKernel::getMaterialAttributes( 
	int index,
	float& r, float& g, float& b, float& noise,
	float& reflection, 
	float& refraction,
	bool&  procedural,
	bool&  wireframe, int& wireframeDepth,
	float& transparency,
	int&   textureId,
	float& specValue, float& specPower, float& specCoef,
   float& innerIllumination, float& illuminationDiffusion, float& illuminationPropagation,
   bool& fastTransparency)
{
	int returnValue = -1;

	if( index>=0 && index<=m_nbActiveMaterials ) 
	{
		r = m_hMaterials[index].color.x;
		g = m_hMaterials[index].color.y;
		b = m_hMaterials[index].color.z;
      noise = m_hMaterials[index].color.w;
		reflection = m_hMaterials[index].reflection.x;
		refraction = m_hMaterials[index].refraction.x;
		transparency = m_hMaterials[index].transparency.x;
		textureId = m_hMaterials[index].textureMapping.z;
		specValue = m_hMaterials[index].specular.x;
		specPower = m_hMaterials[index].specular.y;
		specCoef  = m_hMaterials[index].specular.w;
		innerIllumination = m_hMaterials[index].innerIllumination.x;
		illuminationDiffusion = m_hMaterials[index].innerIllumination.y;
		illuminationPropagation = m_hMaterials[index].innerIllumination.z;
      fastTransparency = (m_hMaterials[index].attributes.x == 1);
		procedural       = (m_hMaterials[index].attributes.y == 1);
		wireframe        = (m_hMaterials[index].attributes.z == 1);
		wireframeDepth   = m_hMaterials[index].attributes.w;
		returnValue = 0;
	}
	else
	{
		LOG_ERROR("GPUKernel::setMaterial: Out of bounds(" << index << "/" << NB_MAX_MATERIALS << ")" );
	}
	return returnValue;
}

Material* GPUKernel::getMaterial( const int index )
{
	Material* returnValue = NULL;

	if( index>=0 && index<=static_cast<int>(m_nbActiveMaterials) ) 
	{
      returnValue = &m_hMaterials[index];
		m_materialsTransfered=false;
   }
   return returnValue;
}


// ---------- Textures ----------
void GPUKernel::setTexture(
   const int index,
	const TextureInformation& textureInfo )
{
   LOG_INFO(3,"GPUKernel::setTexture(" << index << ")" );
   if( index<m_nbActiveTextures )
   {
      delete [] m_hTextures[index].buffer;
      int size = textureInfo.size.x*textureInfo.size.y*textureInfo.size.z;
	   m_hTextures[index].buffer = new unsigned char[size];
      m_hTextures[index].offset = 0;
	   m_hTextures[index].size.x = textureInfo.size.x;
	   m_hTextures[index].size.y = textureInfo.size.y;
	   m_hTextures[index].size.z = textureInfo.size.z;
	   int j(0);
      for( int i(0); i<textureInfo.size.x*textureInfo.size.y*textureInfo.size.z; i+=textureInfo.size.z ) {
         m_hTextures[index].buffer[j]   = textureInfo.buffer[i+2];
		   m_hTextures[index].buffer[j+1] = textureInfo.buffer[i+1];
		   m_hTextures[index].buffer[j+2] = textureInfo.buffer[i];
		   j+=textureInfo.size.z;
	   }
   }
}

void GPUKernel::setSceneInfo(
	int    width,
	int    height,
	float  transparentColor,
	int    shadowsEnabled,
	float  viewDistance,
	float  shadowIntensity,
	int    nbRayIterations,
	float4 backgroundColor,
	int    renderingType,
	float  width3DVision,
	bool   renderBoxes,
	int    pathTracingIteration,
	int    maxPathTracingIterations,
	OutputType outputType,
   int    timer,
   int    fogEffect)
{
	LOG_INFO(3,"GPUKernel::setSceneInfo" );
	memset(&m_sceneInfo,0,sizeof(SceneInfo));
	m_sceneInfo.width.x                = width;
	m_sceneInfo.height.x               = height;
	m_sceneInfo.transparentColor.x     = transparentColor;
	m_sceneInfo.graphicsLevel.x        = shadowsEnabled;
	m_sceneInfo.viewDistance.x         = viewDistance;
	m_sceneInfo.shadowIntensity.x      = shadowIntensity;
	m_sceneInfo.nbRayIterations.x      = nbRayIterations;
	m_sceneInfo.backgroundColor        = backgroundColor;
   m_sceneInfo.renderingType.x        = renderingType;
	m_sceneInfo.width3DVision.x        = width3DVision;
	m_sceneInfo.renderBoxes.x          = renderBoxes;
	m_sceneInfo.pathTracingIteration.x = pathTracingIteration;
	m_sceneInfo.maxPathTracingIterations.x = maxPathTracingIterations;
	m_sceneInfo.misc.x = outputType;
	m_sceneInfo.misc.y = timer;
	m_sceneInfo.misc.z = fogEffect;
}

void GPUKernel::setPostProcessingInfo( 
	PostProcessingType type,
	float              param1,
	float              param2,
	int                param3 )
{
	LOG_INFO(3,"GPUKernel::setPostProcessingInfo" );
	m_postProcessingInfo.type.x   = type;
	m_postProcessingInfo.param1.x = param1;
	m_postProcessingInfo.param2.x = param2;
	m_postProcessingInfo.param3.x = param3;
}

/*
*
*/
char* GPUKernel::loadFromFile( const std::string& filename, size_t& length )
{
	// Load the GPUKernel source code into the array source_str
	FILE *fp = 0;
	char *source_str = 0;

#ifdef WIN32
	fopen_s( &fp, filename.c_str(), "r");
#else
	fp = fopen( filename.c_str(), "r");
#endif
	if( fp == 0 ) 
	{
		LOG_ERROR("Failed to load OpenCL file: " << filename.c_str() );
	}
	else 
	{
		source_str = (char*)malloc(MAX_SOURCE_SIZE);
		length = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
		fclose( fp );
	}

	/*
	for( int i(0); i<MAX_SOURCE_SIZE; ++i)
	{
	char a = source_str[i];
	source_str[i] = a >> 2;
	}
	*/
	return source_str;
}

void GPUKernel::saveToFile( const std::string& filename, const std::string& content )
{
	// Load the GPUKernel source code into the array source_str
	FILE *fp = 0;
	std::string tmp(content);

#ifdef WIN32
	fopen_s( &fp, filename.c_str(), "w");
#else
	fp = fopen( filename.c_str(), "w");
#endif

	/*
	for( int i(0); i<content.length(); ++i)
	{
	char a = tmp[i];
	tmp[i] = a << 2;
	}
	*/

	if( fp == 0 ) 
	{
		LOG_INFO(3,"Failed to save GPUKernel " << filename.c_str() );
	}
	else 
	{
		fwrite( tmp.c_str(), 1, tmp.length(), fp);
		fclose( fp );
	}
}

// ---------- Kinect ----------
bool GPUKernel::loadTextureFromFile( const int index, const std::string& filename )
{
   bool result(false);

   if( filename.length() != 0 )
   {
      m_texturesTransfered = false;
      ImageLoader imageLoader;
      if( filename.find(".bmp") != -1 )
      {
         result = imageLoader.loadBMP24( index, filename, m_hTextures );
      }
      else if ( filename.find(".jpg") != -1 )
      {
         result = imageLoader.loadJPEG( index, filename, m_hTextures );
      }
      else if ( filename.find(".tga") != -1 )
      {
         result = imageLoader.loadTGA( index, filename, m_hTextures );
      }

      if( result )
      {
         m_textureFilenames[index] = filename;
         ++m_nbActiveTextures;
      }
      else
      {
         LOG_ERROR("Failed to load " << filename );
      }
   }
   return result;
}

void GPUKernel::buildLightInformationFromTexture( unsigned int index )
{
   m_lightInformationSize = 0;

   // Light from explicit light sources
   for( int i(0); i<m_nbActiveLamps[m_frame]; ++i )
   {
      int idx((*m_lamps[m_frame])[i]);
      CPUPrimitive& lamp = (*m_primitives[m_frame])[idx];
      Material& material = m_hMaterials[lamp.materialId];

      LightInformation lightInformation;
      lightInformation.attribute.x = idx;

      lightInformation.location.x = lamp.p0.x;
      lightInformation.location.y = lamp.p0.y;
      lightInformation.location.z = lamp.p0.z;
      
      lightInformation.color.x = material.color.x;
      lightInformation.color.y = material.color.y;
      lightInformation.color.z = material.color.z;
      lightInformation.color.w = material.innerIllumination.x;

      m_lightInformation[m_lightInformationSize] = lightInformation;

      LOG_INFO(3,
         "Lamp " << m_lightInformation[m_lightInformationSize].attribute.x << ":" <<
         m_lightInformation[m_lightInformationSize].location.x << "," <<
         m_lightInformation[m_lightInformationSize].location.y << "," <<
         m_lightInformation[m_lightInformationSize].location.z << " " <<
         m_lightInformation[m_lightInformationSize].color.x << "," <<
         m_lightInformation[m_lightInformationSize].color.y << "," <<
         m_lightInformation[m_lightInformationSize].color.z << " " <<
         m_lightInformation[m_lightInformationSize].color.w );

      m_lightInformationSize++;
   }
   float size = m_sceneInfo.viewDistance.x/3.f;

   float pi = static_cast<float>(PI);
#if 0
   float x = 0.f;
   for( int i(0); i<m_sceneInfo.maxPathTracingIterations.x/10; ++i)
   {
      LightInformation lightInformation;
      lightInformation.location.x = rand()%10000 - 5000.f;
      lightInformation.location.y = rand()%10000 - 5000.f;
      lightInformation.location.z = rand()%10000 - 5000.f;
      lightInformation.attribute.x = -1;
      lightInformation.color.x = 0.5f+0.5f*cos(x);
      lightInformation.color.y = 0.5f+0.5f*sin(x);
      lightInformation.color.z = 0.5f+0.5f*cos(x+pi);
      lightInformation.color.w = 0.2f;
      m_lightInformation[m_lightInformationSize] = lightInformation;
      m_lightInformationSize++;
	  x += (2.f*pi)/static_cast<float>(m_sceneInfo.maxPathTracingIterations.x/10);
   }
#else
   /*
   for( float x(0.f); x<2.f*M_PI; x+=M_PI/8.f)
   {
      LightInformation lightInformation;
      lightInformation.location.x = 200.f*cos(x);
      lightInformation.location.y = 200.f*sin(x);
      lightInformation.location.z = -2000.f;
      lightInformation.attribute.x = -1;
      lightInformation.color.x = 0.5f+0.5f*cos(x);
      lightInformation.color.y = 0.5f+0.5f*sin(x);
      lightInformation.color.z = 0.5f+0.5f*cos(x+pi);
      lightInformation.color.w = 0.2f;
      m_lightInformation[m_lightInformationSize] = lightInformation;
      m_lightInformationSize++;
   }
   */

   /*
   // Light from skybox
   if( index < m_textureIndex )
   {
      for( int i(0); i<gTextureWidth*gTextureHeight; i+=gTextureDepth*4)
      {
         int Y = i/gTextureHeight;
         int Z = i%gTextureWidth;

         int  offset = gTextureOffset + (index*gTextureWidth*gTextureHeight*gTextureDepth);
         unsigned char r = m_hTextures[offset+i];
         unsigned char g = m_hTextures[offset+i+1];
         unsigned char b = m_hTextures[offset+i+2];
         float R = r/255.f;
         float G = g/255.f;
         float B = b/255.f;

         if( (R+G+B)>1.f )
         {
            float4 lampLocation;
            lampLocation.x = 10.f*static_cast<float>(-gTextureWidth/2  + Z);
            lampLocation.y = 10.f*static_cast<float>(-gTextureHeight/2 + Y);
            lampLocation.z = -10.f*size;
            lampLocation.w = -1.f;
            float4 lampInformation;
            lampInformation.x = R;
            lampInformation.y = G;
            lampInformation.z = B;
            lampInformation.w = 1.f;
            m_lightInformation[m_lightInformationSize].location    = lampLocation;
            m_lightInformation[m_lightInformationSize].information = lampInformation;

            LOG_INFO(3,
               "Lamp " << 
               m_lightInformation[m_lightInformationSize].location.x << "," <<
               m_lightInformation[m_lightInformationSize].location.y << "," <<
               m_lightInformation[m_lightInformationSize].location.z << " " <<
               m_lightInformation[m_lightInformationSize].information.x << "," <<
               m_lightInformation[m_lightInformationSize].information.y << "," <<
               m_lightInformation[m_lightInformationSize].information.z << " " <<
               m_lightInformation[m_lightInformationSize].information.w );

            m_lightInformationSize++;
         }
      }
   }
   */
#endif // 0
   LOG_INFO(3, "Light Information Size = " << m_nbActiveLamps << "/" << m_lightInformationSize );
}

#ifdef USE_KINECT
void GPUKernel::initializeKinectTextures()
{
   LOG_INFO(1, "Initializing Kinect textures" );
   m_hTextures[0].offset = 0;
   m_hTextures[0].size.x = gKinectVideoWidth;
   m_hTextures[0].size.y = gKinectVideoHeight;
   m_hTextures[0].size.z = gKinectVideo;

   m_hTextures[1].offset = gKinectVideoSize;
   m_hTextures[1].size.x = gKinectDepthWidth;
   m_hTextures[1].size.y = gKinectDepthHeight;
   m_hTextures[1].size.z = gKinectDepth;

   m_nbActiveTextures = 2;
}

int GPUKernel::updateSkeletons( 
	unsigned int primitiveIndex, 
	float3 skeletonPosition, 
	float size,
	float radius,       int materialId,
	float head_radius,  int head_materialId,
	float hands_radius, int hands_materialId,
	float feet_radius,  int feet_materialId)
{
	m_skeletonIndex = -1;
	HRESULT hr = NuiSkeletonGetNextFrame( 0, &m_skeletonFrame );
	bool found = false;
	if( hr == S_OK )
	{
		int i=0;
		while( i<NUI_SKELETON_COUNT && !found ) 
		{
			if( m_skeletonFrame.SkeletonData[i].eTrackingState == NUI_SKELETON_TRACKED ) 
			{
				m_skeletonIndex = i;
				found = true; //(m_skeletonIndex==0);
				/*
				for( int j=0; j<20; j++ ) 
				{
				float r = radius;
				int   m = materialId;
				switch (j) {
				case NUI_SKELETON_POSITION_FOOT_LEFT:
				case NUI_SKELETON_POSITION_FOOT_RIGHT:
				r = feet_radius;
				m = feet_materialId;
				break;
				case NUI_SKELETON_POSITION_HAND_LEFT:
				case NUI_SKELETON_POSITION_HAND_RIGHT:
				r = hands_radius;
				m = hands_materialId;
				break;
				case NUI_SKELETON_POSITION_HEAD:
				r = head_radius;
				m = head_materialId;
				break;
				}
				setPrimitive(
				primitiveIndex+j,
				boxId,
				static_cast<float>( m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].x * size + skeletonPosition.x),
				static_cast<float>( m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].y * size + skeletonPosition.y),
				static_cast<float>( skeletonPosition.z - 2.f*m_skeletonFrame.SkeletonData[i].SkeletonPositions[j].z * size ),
				static_cast<float>(r), 
				static_cast<float>(r), 
				m,
				1, 1 );
				}
				*/
			}
			i++;
		}
	}
	return found ? S_OK : S_FALSE;
}

bool GPUKernel::getSkeletonPosition( int index, float3& position )
{
	bool returnValue(false);
	if( m_skeletonIndex != -1 ) 
	{
		position.x = m_skeletonFrame.SkeletonData[m_skeletonIndex].SkeletonPositions[index].x;
		position.y = m_skeletonFrame.SkeletonData[m_skeletonIndex].SkeletonPositions[index].y;
		position.z = m_skeletonFrame.SkeletonData[m_skeletonIndex].SkeletonPositions[index].z;
		returnValue = true;
	}
	return returnValue;
}

#endif // USE_KINECT

void GPUKernel::saveBitmapToFile( const std::string& filename, BitmapBuffer* bitmap, const int width, const int height, const int depth )
{
	FILE *f;

	unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
	unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 32,0};
	unsigned char bmppad[3] = {0,0,0};

	int w = width;
	int h = height;
	int filesize = 54 + depth*w*h;

	bmpfileheader[ 2] = (unsigned char)(filesize    );
	bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
	bmpfileheader[ 4] = (unsigned char)(filesize>>16);
	bmpfileheader[ 5] = (unsigned char)(filesize>>24);

	bmpinfoheader[ 4] = (unsigned char)(       w    );
	bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
	bmpinfoheader[ 6] = (unsigned char)(       w>>16);
	bmpinfoheader[ 7] = (unsigned char)(       w>>24);

	bmpinfoheader[ 8] = (unsigned char)(       h    );
	bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
	bmpinfoheader[10] = (unsigned char)(       h>>16);
	bmpinfoheader[11] = (unsigned char)(       h>>24);

#ifdef WIN32
	fopen_s(&f, filename.c_str(),"wb");
#else
	f = fopen(filename.c_str(),"wb");
#endif 
	fwrite(bmpfileheader,1,14,f);
	fwrite(bmpinfoheader,1,40,f);
	fwrite(bitmap,1,width*height*depth,f );
	fclose(f);
};

int GPUKernel::getLight( int index )
{
   if( index < m_nbActiveLamps[m_frame] )
   {
      return (*m_lamps[m_frame])[index];
   }
   return -1;
}

// OpenGL
int GPUKernel::setGLMode( const int& glMode )
{
   int p=-1;
   int frame(0);
   if( glMode == -1 )
   {
      switch( m_GLMode )
      {
      case GL_TRIANGLES:
         {
            // Vertices
            if( m_vertices.size()%3 == 0 )
            {
               int nbTriangles=m_vertices.size()/3;
               for( int i(0); i<nbTriangles; ++i)
               {
                  p = addPrimitive( ptTriangle);
                  setPrimitive(  p,
                     m_vertices[i*3+0].x, m_vertices[i*3+0].y, m_vertices[i*3+0].z, 
                     m_vertices[i*3+1].x, m_vertices[i*3+1].y, m_vertices[i*3+1].z, 
                     m_vertices[i*3+2].x, m_vertices[i*3+2].y, m_vertices[i*3+2].z, 
                     0.f,0.f,0.f,
                     0);

                  if( m_textCoords.size()>=i*3+2 )
                  {
                     setPrimitiveTextureCoordinates(  p, m_textCoords[i*3+0], m_textCoords[i*3+1], m_textCoords[i*3+2] );
                  }
                  if( m_normals.size()>=i*3+2 )
                  {
                     setPrimitiveNormals(  p, m_normals[i*3+0], m_normals[i*3+1], m_normals[i*3+2] );
                  }
               }
               LOG_INFO(3, "Triangle created");
            }
            else
            {
               LOG_ERROR("Incorrect number of vertices to create a triangle" );
            }


         }
         break;
      }
      m_vertices.clear();
      m_normals.clear();
      m_textCoords.clear();
   }
   m_GLMode = glMode;
   return p;
}

void GPUKernel::addVertex( float x, float y, float z)
{
   float3 v = {x,y,z};
   m_vertices.push_back(v);
}

void GPUKernel::addNormal( float x, float y, float z)
{
   float3 v = {x,y,z};
   m_normals.push_back(v);
}

void GPUKernel::addTextCoord( float x, float y, float z)
{
   float3 v = {x,y,z};
   m_textCoords.push_back(v);
}
