#ifdef WIN32
#include <windows.h>
#else
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#endif

#include <iostream>
#include <vector>

#include "GPUKernel.h"
#include "Logging.h"
#include "Consts.h"

const unsigned int MAX_SOURCE_SIZE = 65535;
const unsigned int OPTIMAL_NB_OF_PRIMITIVES_PER_BOXES = 250;

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

GPUKernel::GPUKernel(bool activeLogging, int platform, int device)
 : m_hPrimitives(0), 
	m_hMaterials(0), 
	m_hTextures(0),
   m_hLamps(0),
   m_hBoundingBoxes(0),
	m_hDepthOfField(0),
   m_hPrimitivesXYIds(0),
	m_hRandoms(0), 
	m_nbActiveLamps(0),
	m_nbActiveMaterials(-1),
	m_nbActiveTextures(0),
	m_activeLogging(activeLogging),
#if USE_KINECT
	m_hVideo(0), m_hDepth(0), 
	m_skeletons(0), m_hNextDepthFrameEvent(0), m_hNextVideoFrameEvent(0), m_hNextSkeletonEvent(0),
	m_pVideoStreamHandle(0), m_pDepthStreamHandle(0),
	m_skeletonsBody(-1), m_skeletonsLamp(-1), m_skeletonIndex(-1),
#endif // USE_KINECT
	m_textureTransfered(false)
{
	LOG_INFO(3,"GPUKernel::GPUKernel (Log is " << (activeLogging ? "" : "de") << "activated" );

#if 1
	std::cout <<"CPU: SceneInfo         : " << sizeof(SceneInfo) << std::endl;
	std::cout <<"CPU: Ray               : " << sizeof(Ray) << std::endl;
	std::cout <<"CPU: PrimitiveType     : " << sizeof(PrimitiveType) << std::endl;
	std::cout <<"CPU: Material          : " << sizeof(Material) << std::endl;
	std::cout <<"CPU: BoundingBox       : " << sizeof(BoundingBox) << std::endl;
	std::cout <<"CPU: Primitive         : " << sizeof(Primitive) << std::endl;
	std::cout <<"CPU: PostProcessingType: " << sizeof(PostProcessingType) << std::endl;
	std::cout <<"CPU: PostProcessingInfo: " << sizeof(PostProcessingInfo) << std::endl;
	std::cout <<"Textures " << NB_MAX_TEXTURES << std::endl;
#endif // 0
}


GPUKernel::~GPUKernel()
{
	LOG_INFO(3,"GPUKernel::~GPUKernel");
}

void GPUKernel::initBuffers()
{
	LOG_INFO(3,"GPUKernel::initBuffers");

	// Setup World
	m_hPrimitives = new Primitive[NB_MAX_PRIMITIVES];
	memset(m_hPrimitives,0,NB_MAX_PRIMITIVES*sizeof(Primitive) ); 
	m_hMaterials = new Material[NB_MAX_MATERIALS+1];
	memset(m_hMaterials,0,NB_MAX_MATERIALS*sizeof(Material) ); 
	m_hTextures = new char[gTextureOffset+gTextureSize*NB_MAX_TEXTURES];
	m_hBoundingBoxes = new BoundingBox[NB_MAX_BOXES];
	memset(m_hBoundingBoxes,0,NB_MAX_BOXES*sizeof(BoundingBox));
	m_hLamps = new int[NB_MAX_LAMPS];
	memset(m_hLamps,0,NB_MAX_LAMPS*sizeof(int));

	// Randoms
	int size = m_sceneInfo.width.x*m_sceneInfo.height.x;

	m_hPrimitivesXYIds = new int[size];
	memset( m_hPrimitivesXYIds,0,size*sizeof(int));

	m_hRandoms = new float[size];
	int i;
#pragma omp parallel for
	for( i=0; i<size; ++i)
	{
		m_hRandoms[i] = (rand()%2000-1000)/1000.f;
	}
}

void GPUKernel::cleanup()
{
	if(m_hPrimitives!=0) delete m_hPrimitives;
	if(m_hMaterials!=0) delete m_hMaterials;
	if(m_hTextures!=0) delete m_hTextures;
	if(m_hBoundingBoxes!=0) delete m_hBoundingBoxes;
	if(m_hLamps!=0) delete m_hLamps;
	if(m_hPrimitivesXYIds!=0) delete m_hPrimitivesXYIds;
	if(m_hRandoms!=0) delete m_hRandoms;

   m_boundingBoxes.clear();
   m_primitives.clear();
   m_hPrimitives = 0; 
	m_hMaterials = 0;
	m_hTextures = 0;
	m_hDepthOfField = 0;
	m_hRandoms = 0;
   m_hLamps = 0,
   m_hBoundingBoxes = 0,
   m_hPrimitivesXYIds = 0,
	m_nbActiveLamps = 0;
	m_nbActiveMaterials = -1;
	m_nbActiveTextures = 0;
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
	m_textureTransfered = false;

   m_minPos.x =  100000.f;
   m_minPos.y =  100000.f;
   m_minPos.z =  100000.f;
   m_maxPos.x = -100000.f;
   m_maxPos.y = -100000.f;
   m_maxPos.z = -100000.f;
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
}

int GPUKernel::addPrimitive( PrimitiveType type )
{
	LOG_INFO(3,"GPUKernel::addPrimitive");
   CPUPrimitive primitive;
   memset(&primitive,0,sizeof(CPUPrimitive));
   primitive.type = type;
   m_primitives[static_cast<int>(m_primitives.size())] = primitive;
   LOG_INFO(3,"m_primitives.size() = " << m_primitives.size());
	return static_cast<int>(m_primitives.size()-1);
}

CPUPrimitive* GPUKernel::getPrimitive( const unsigned int index )
{
   CPUPrimitive* returnValue(NULL);
	if( index>=0 && index<=m_primitives.size()) 
   {
      returnValue = &m_primitives[index];
   }
   return returnValue;
}

void GPUKernel::setPrimitive( 
	unsigned int index,
	float x0, float y0, float z0, 
	float w,  float h,  float d,
	int   materialId, 
	float materialPaddingX, float materialPaddingY )
{
	setPrimitive( index, x0, y0, z0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, materialId, materialPaddingX, materialPaddingY );
}

void GPUKernel::setPrimitive( 
	unsigned int index,
	float x0, float y0, float z0, 
	float x1, float y1, float z1, 
	float w,  float h,  float d,
	int   materialId, 
	float materialPaddingX, float materialPaddingY )
{
	setPrimitive( index, x0, y0, z0, x1, y1, z1, 0.f, 0.f, 0.f, w, h, d, materialId, materialPaddingX, materialPaddingY );
}

void GPUKernel::setPrimitive( 
	unsigned int index,
	float x0, float y0, float z0, 
	float x1, float y1, float z1, 
	float x2, float y2, float z2, 
	float w,  float h,  float d,
	int   materialId, 
	float materialPaddingX, float materialPaddingY )
{
   /*
	LOG_INFO(3,"GPUKernel::setPrimitive( " << 
		index << " (" << 
		"center (" << x0 << "," << y0 << "," << z0 << ")," <<
		"size (" << w << "," << h << "," << d << ")," <<
		"material (" << materialId << "," << materialPaddingX << "," << materialPaddingY << ")"
		);
      */
   if( index>=0 && index<m_primitives.size()) 
	{
      m_primitives[index].movable= true;
		m_primitives[index].p0.x   = x0;
		m_primitives[index].p0.y   = y0;
		m_primitives[index].p0.z   = z0;
		m_primitives[index].p1.x   = x1;
		m_primitives[index].p1.y   = y1;
		m_primitives[index].p1.z   = z1;
		m_primitives[index].p2.x   = x2;
		m_primitives[index].p2.y   = y2;
		m_primitives[index].p2.z   = z2;
		m_primitives[index].size.x = w;
		m_primitives[index].size.y = h;
		m_primitives[index].size.z = d;
		m_primitives[index].materialId = materialId;

		switch( m_primitives[index].type )
		{
		case ptSphere:
         {
			   m_primitives[index].size.x = w;
			   m_primitives[index].size.y = w;
			   m_primitives[index].size.z = w;
			   m_primitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
			   m_primitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
			   break;
         }
		case ptEllipsoid:
			{
		      m_primitives[index].size.x = w;
		      m_primitives[index].size.y = h;
		      m_primitives[index].size.z = d;
			   m_primitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
			   m_primitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
            break;
         }
		case ptCylinder:
			{
				// Axis
				float3 axis;
				axis.x = x1 - x0;
				axis.y = y1 - y0;
				axis.z = z1 - z0;
				float len = sqrt( axis.x*axis.x + axis.y*axis.y + axis.z*axis.z );
            if( len != 0.f )
            {
				   axis.x /= len;
				   axis.y /= len;
				   axis.z /= len;
            }
				m_primitives[index].n1.x = axis.x;
				m_primitives[index].n1.y = axis.y;
				m_primitives[index].n1.z = axis.z;

            m_primitives[index].size.x = w; //(x1 - x0)/2.f;
            m_primitives[index].size.y = w; //(y1 - y0)/2.f;
            m_primitives[index].size.z = w; // (z1 - z0)/2.f;

				// Material
				m_primitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
				m_primitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
				break;
			}
#ifdef USE_KINECT 
		case ptCamera:
			{
				m_primitives[index].n0.x = 0.f;
				m_primitives[index].n0.y = 0.f;
				m_primitives[index].n0.z = -1.f;
				m_primitives[index].materialInfo.x = ( w != 0.f ) ? (gKinectVideoWidth /w/2)*materialPaddingX : 1.f;
				m_primitives[index].materialInfo.y = ( w != 0.f ) ? (gKinectVideoHeight/h/2)*materialPaddingY : 1.f;
				break;
			}
#endif // USE_KINECT
		case ptXYPlane:
			{
				m_primitives[index].n0.x = 0.f;
				m_primitives[index].n0.y = 0.f;
				m_primitives[index].n0.z = 1.f;
				m_primitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
				m_primitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
				break;
			}
		case ptYZPlane:
			{
				m_primitives[index].n0.x = 1.f;
				m_primitives[index].n0.y = 0.f;
				m_primitives[index].n0.z = 0.f;
				m_primitives[index].materialInfo.x = ( d != 0.f ) ? (gTextureWidth /d/2)*materialPaddingX : 1.f;
				m_primitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
				break;
			}
		case ptXZPlane:
		case ptCheckboard:
			{
				m_primitives[index].n0.x = 0.f;
				m_primitives[index].n0.y = 1.f;
				m_primitives[index].n0.z = 0.f;
				m_primitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
				m_primitives[index].materialInfo.y = ( d != 0.f ) ? (gTextureHeight/d/2)*materialPaddingY : 1.f;
				break;
			}
		case ptTriangle:
			{
            float3 v0,v1;
            v0.x = m_primitives[index].p1.x-m_primitives[index].p0.x;
            v0.y = m_primitives[index].p1.y-m_primitives[index].p0.y;
            v0.z = m_primitives[index].p1.z-m_primitives[index].p0.z;
            normalizeVector(v0);
            
            v1.x = m_primitives[index].p2.x-m_primitives[index].p0.x;
            v1.y = m_primitives[index].p2.y-m_primitives[index].p0.y;
            v1.z = m_primitives[index].p2.z-m_primitives[index].p0.z;
            normalizeVector(v1);

            m_primitives[index].n0 = crossProduct(v0,v1);
            normalizeVector(m_primitives[index].n0);
				break;
			}
		}
      //min
      m_minPos.x = std::min(x0,m_minPos.x);
      m_minPos.y = std::min(y0,m_minPos.y);
      m_minPos.z = std::min(z0,m_minPos.z);
             
      // max
      m_maxPos.x = std::max(x0,m_maxPos.x);
      m_maxPos.y = std::max(y0,m_maxPos.y);
      m_maxPos.z = std::max(z0,m_maxPos.z);
   }
	else
	{
		LOG_ERROR("GPUKernel::setPrimitive: Out of bounds (" << index << "/" << NB_MAX_PRIMITIVES << ")" );
	}
}

void GPUKernel::setPrimitiveIsMovable( int unsigned index, bool movable )
{
   if( index>=0 && index<m_primitives.size()) 
	{
      CPUPrimitive& primitive(m_primitives[index]);
      primitive.movable = movable;
   }
}


void GPUKernel::setPrimitiveNormals( int unsigned index, float3 n0, float3 n1, float3 n2 )
{
   if( index>=0 && index<m_primitives.size()) 
	{
      CPUPrimitive& primitive(m_primitives[index]);
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
		returnValue = m_hPrimitivesXYIds[index];
	return returnValue;
}

void GPUKernel::updateBoundingBox( CPUBoundingBox& box )
{
	LOG_INFO(3,"GPUKernel::updateBoundingBox()" );

   // Process box size
	float3 corner0;
	float3 corner1;

   std::vector<unsigned int>::const_iterator it = box.primitives.begin();
   while( it != box.primitives.end() )
   {
      CPUPrimitive& primitive = m_primitives[*it];
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
}

void GPUKernel::resetBoxes( bool resetPrimitives )
{
   if( resetPrimitives )
   {
      for( int i(0); i<m_boundingBoxes.size(); ++i )
      {
         resetBox( m_boundingBoxes[i], resetPrimitives );
      }
   }
   else
   {
      m_boundingBoxes.clear();
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

unsigned int GPUKernel::processBoxes( const unsigned int boxSize, unsigned int& nbActiveBoxes, bool simulate )
{
   float3 boxSteps;
   boxSteps.x = ( m_maxPos.x - m_minPos.x ) / boxSize;
   boxSteps.y = ( m_maxPos.y - m_minPos.y ) / boxSize;
   boxSteps.z = ( m_maxPos.z - m_minPos.z ) / boxSize;
   //std::cout << "Steps " << boxSteps.x << "," << boxSteps.y << "," << boxSteps.z  << std::endl;

   boxSteps.x = ( boxSteps.x == 0.f ) ? 1 : boxSteps.x;
   boxSteps.y = ( boxSteps.y == 0.f ) ? 1 : boxSteps.y;
   boxSteps.z = ( boxSteps.z == 0.f ) ? 1 : boxSteps.z;

   //std::cout << "min " << m_minPos.x << "," << m_minPos.y << "," << m_minPos.z  << std::endl;
   //std::cout << "max " << m_maxPos.x << "," << m_maxPos.y << "," << m_maxPos.z  << std::endl;
   //std::cout << "box [" << boxSize << "] " << boxSteps.x << "," << boxSteps.y << "," << boxSteps.z  << std::endl;

   nbActiveBoxes = 0;
   std::map<unsigned int,unsigned int> primitivesPerBox;

   int p(0);
   std::map<unsigned int,CPUPrimitive>::iterator it = m_primitives.begin();
   while( it != m_primitives.end() )
   {
      CPUPrimitive& primitive((*it).second);

      int X = static_cast<int>(( primitive.p0.x - m_minPos.x ) / boxSteps.x);
      int Y = static_cast<int>(( primitive.p0.y - m_minPos.y ) / boxSteps.y);
      int Z = static_cast<int>(( primitive.p0.z - m_minPos.z ) / boxSteps.z);
      int B = X*boxSize*boxSize + Y*boxSize + Z;

      if( simulate )
      {
         if( primitivesPerBox.find(B) == primitivesPerBox.end() )
         {
            //std::cout << B << std::endl;
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
	         m_boundingBoxes[B].parameters[0].x = m_sceneInfo.viewDistance.x;
	         m_boundingBoxes[B].parameters[0].y = m_sceneInfo.viewDistance.x;
	         m_boundingBoxes[B].parameters[0].z = m_sceneInfo.viewDistance.x;
	         m_boundingBoxes[B].parameters[1].x = -m_sceneInfo.viewDistance.x;
	         m_boundingBoxes[B].parameters[1].y = -m_sceneInfo.viewDistance.x;
	         m_boundingBoxes[B].parameters[1].z = -m_sceneInfo.viewDistance.x;
         }
         m_boundingBoxes[B].primitives.push_back(p);
      }
      ++p;
      ++it;
   }

   int maxPrimitivePerBox(0);
   int delta = 0;
   if( simulate )
   {
#if 0
      std::map<int,int>::const_iterator itpb = primitivesPerBox.begin();
      while( itpb != primitivesPerBox.end() )
      {
         //std::cout << "Box " << (*itpb).first << " -> " << (*itpb).second << std::endl;
         maxPrimitivePerBox = ((*itpb).second>maxPrimitivePerBox) ? (*itpb).second : maxPrimitivePerBox;
         ++itpb;
      }
      //if( primitivesPerBox.size() != 0 ) maxPrimitivePerBox /= static_cast<int>(primitivesPerBox.size());
#else
      maxPrimitivePerBox = static_cast<int>(m_primitives.size()/nbActiveBoxes);
#endif
      delta = OPTIMAL_NB_OF_PRIMITIVES_PER_BOXES-nbActiveBoxes;
      //std::cout << "NbMaxPrimitivePerBox[" << boxSize << "], nbBoxes=" << nbActiveBoxes << ", maxPrimitivePerBox=" << maxPrimitivePerBox << ", Ratio=" << abs(delta) << "/" << OPTIMAL_NB_OF_PRIMITIVES_PER_BOXES << std::endl;
   }
   else
   {
      std::map<unsigned int,CPUBoundingBox>::iterator itb = m_boundingBoxes.begin();
      while( itb != m_boundingBoxes.end() )
      {
         updateBoundingBox((*itb).second);
         itb++;
      }
   }

   return delta;
}

int GPUKernel::compactBoxes( bool reconstructBoxes )
{
	LOG_INFO(3,"GPUKernel::compactBoxes" );

   try 
   {
      if( reconstructBoxes )
      {
         // Bounding boxes
         // Search for best trade-off
         std::map<unsigned int,unsigned int> primitivesPerBox;
         unsigned int maxPrimitivePerBox(0);
         unsigned int boxSize = 64;
         unsigned int bestSize = boxSize;
         unsigned int bestActiveBoxes = 0;
         unsigned int bestRatio = 100000;
         unsigned int activeBoxes(NB_MAX_BOXES);
         do 
         {
            unsigned int ratio = processBoxes( boxSize, activeBoxes, true );
            if( ratio < bestRatio ) 
            {
               bestSize = boxSize;
               bestRatio = ratio;
               bestActiveBoxes = activeBoxes;
            }
            boxSize--;
         }
         while( boxSize>0 );
         //std::cout << "Best trade off: " << bestSize << "/" << bestActiveBoxes << " boxes" << std::endl;
         processBoxes( bestSize, activeBoxes, false );
      }

      unsigned int b(0);
      unsigned int primitivesIndex(0);
      m_nbActivePrimitives = 0;
      m_nbActiveLamps = 0;
      std::map<unsigned int,CPUBoundingBox>::iterator itb=m_boundingBoxes.begin();
      while( itb != m_boundingBoxes.end() )
      {
         CPUBoundingBox& box = (*itb).second;
         if( box.primitives.size() != 0 && b<NB_MAX_BOXES )
         {
            // Prepare boxes for GPU
            m_hBoundingBoxes[b].parameters[0] = box.parameters[0];
            m_hBoundingBoxes[b].parameters[1] = box.parameters[1];
            m_hBoundingBoxes[b].startIndex.x  = primitivesIndex;
            m_hBoundingBoxes[b].nbPrimitives.x= static_cast<int>(box.primitives.size());

            std::vector<unsigned int>::const_iterator itp = box.primitives.begin();
            while( itp != box.primitives.end() )
            {
               // Prepare primitives for GPU
               if( *itp < NB_MAX_PRIMITIVES )
               {
                  CPUPrimitive& primitive = m_primitives[*itp];
                  m_hPrimitives[primitivesIndex].index.x = *itp;
                  m_hPrimitives[primitivesIndex].type.x  = primitive.type;
                  m_hPrimitives[primitivesIndex].p0 = primitive.p0;
                  m_hPrimitives[primitivesIndex].p1 = primitive.p1;
                  m_hPrimitives[primitivesIndex].p2 = primitive.p2;
                  m_hPrimitives[primitivesIndex].n0 = primitive.n0;
                  m_hPrimitives[primitivesIndex].n1 = primitive.n1;
                  m_hPrimitives[primitivesIndex].n2 = primitive.n2;
                  m_hPrimitives[primitivesIndex].size = primitive.size;
                  m_hPrimitives[primitivesIndex].materialId.x = primitive.materialId;
                  m_hPrimitives[primitivesIndex].materialInfo = primitive.materialInfo;

                  // Lights
		            if( primitive.materialId >= 0 && m_hMaterials[primitive.materialId].innerIllumination.x != 0.f )
		            {
                     primitive.movable = false;
			            m_hLamps[m_nbActiveLamps] = primitivesIndex;
			            m_nbActiveLamps++;
			            LOG_INFO(3,"Lamp added (" << m_nbActiveLamps << "/" << NB_MAX_LAMPS << ")" );
		            }
               }
               ++itp;
               ++primitivesIndex;
            }
            ++b;
         }
         ++itb;
      }
      //std::cout << "Compacted boxes (" << b << "/" << m_boundingBoxes.size() << ")" << std::endl;
      m_nbActivePrimitives=primitivesIndex;
   }
   catch( ... ) 
   {
   }
	return static_cast<int>(m_boundingBoxes.size());
}

void GPUKernel::displayBoxesInfo()
{
	for( unsigned int i(0); i<=m_boundingBoxes.size(); ++i )
	{
		CPUBoundingBox& box = m_boundingBoxes[i];
		std::cout << "Box " << i << std::endl;
		std::cout << "- # of primitives: " << box.primitives.size() << std::endl;
		std::cout << "- Corners 1      : " << box.parameters[0].x << "," << box.parameters[0].y << "," << box.parameters[0].z << std::endl;
		std::cout << "- Corners 2      : " << box.parameters[1].x << "," << box.parameters[1].y << "," << box.parameters[1].z << std::endl;
      unsigned int p(0);
      std::vector<unsigned int>::const_iterator it = box.primitives.begin();
      while( it != box.primitives.end() )
      {
         CPUPrimitive& primitive(m_primitives[*it]);
			std::cout << "- - " << 
				p << ":" << 
				"type = " << primitive.type << ", " << 
				"center = (" << primitive.p0.x << "," << primitive.p0.y << "," << primitive.p0.z << "), " << 
				"p1 = (" << primitive.p1.x << "," << primitive.p1.y << "," << primitive.p1.z << ")" <<
				std::endl;
         ++p;
         ++it;
		}
		std::cout << std::endl;
	}
}

void GPUKernel::rotatePrimitives( float3 rotationCenter, float3 angles, unsigned int from, unsigned int to )
{
	LOG_INFO(3,"GPUKernel::rotatePrimitives(" << from << "->" << to << ")" );
	float3 cosAngles, sinAngles;
	cosAngles.x = cos(angles.x);
	cosAngles.y = cos(angles.y);
	cosAngles.z = cos(angles.z);
	sinAngles.x = sin(angles.x);
	sinAngles.y = sin(angles.y);
	sinAngles.z = sin(angles.z);

#if 0
	int i=0;
#pragma omp parallel for
	for( i=0; i<m_boundingBoxes.size(); ++i )
	{
      std::map<int,CPUBoundingBox>::iterator itb = m_boundingBoxes.find(i);
      if( itb != m_boundingBoxes.end() )
      {
         CPUBoundingBox& box((*itb).second);
         resetBox(box, false);
         std::vector<int>::const_iterator it = box.primitives.begin();
         while( it != box.primitives.end() )
		   {
            CPUPrimitive& primitive(m_primitives[*it]);
			   rotatePrimitive( primitive, rotationCenter, cosAngles, sinAngles );
            ++it;
		   }
         updateBoundingBox(box);
      }
	}
#else
   std::map<unsigned int,CPUBoundingBox>::iterator itb = m_boundingBoxes.begin();
   while( itb != m_boundingBoxes.end() )
   {
      CPUBoundingBox& box((*itb).second);
      resetBox(box,false);
      std::vector<unsigned int>::const_iterator it = box.primitives.begin();
      while( it != box.primitives.end() )
		{
         CPUPrimitive& primitive(m_primitives[*it]);
         if( primitive.movable && primitive.type != ptCamera )
         {
			   rotatePrimitive( primitive, rotationCenter, cosAngles, sinAngles );
         } 
         ++it;
		}
      updateBoundingBox(box);
      ++itb;
   }
#endif // 0

   compactBoxes(false);
}

void GPUKernel::rotateVector( float3& v, const float3 rotationCenter, const float3& cosAngles, const float3& sinAngles )
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
         rotateVector( primitive.n0, rotationCenter, cosAngles, sinAngles );
         if( primitive.type == ptTriangle )
         {
            rotateVector( primitive.n1, rotationCenter, cosAngles, sinAngles );
            rotateVector( primitive.n2, rotationCenter, cosAngles, sinAngles );
         }
         else
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

void GPUKernel::getPrimitiveCenter(
	unsigned int index, 
	float& x, 
	float& y, 
	float& z,
	float& w)
{
	if( index>=0 && index<=m_primitives.size()) 
	{
		x = m_primitives[index].p0.x;
		y = m_primitives[index].p0.y;
		z = m_primitives[index].p0.z;
	}
}

void GPUKernel::getPrimitiveOtherCenter(
	unsigned int index, 
	float3& p1)
{
	if( index>=0 && index<=m_primitives.size()) 
	{
		p1 = m_primitives[index].p1;
	}
}

void GPUKernel::setPrimitiveCenter(
	unsigned int index, 
	float x, 
	float y, 
	float z,
	float w)
{
   // TODO, Box needs to be updated
	if( index>=0 && index<=m_primitives.size()) 
	{
		m_primitives[index].p0.x = x;
		m_primitives[index].p0.y = y;
		m_primitives[index].p0.z = z;
		m_primitives[index].size.x = w;
	}
}

int GPUKernel::addCube( 
	unsigned int boxId,
	float x, float y, float z, 
	float radius, 
	int   materialId, 
	float materialPaddingX, float materialPaddingY )
{
	LOG_INFO(3,"GPUKernel::addCube(" << boxId << ")" );
	return addRectangle(boxId, x,y,z,radius,radius,radius,materialId,materialPaddingX,materialPaddingY);
}

int GPUKernel::addRectangle( 
	unsigned int boxId,
	float x, float y, float z, 
	float w, float h, float d,
	int   materialId, 
	float materialPaddingX, float materialPaddingY )
{
	LOG_INFO(3,"GPUKernel::addRectangle(" << boxId << ")" );
	int returnValue;
	// Back
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, x, y, z+d, w, h, d, materialId, materialPaddingX, materialPaddingY ); 

	// Front
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, x, y, z-d, w, h, d, materialId, materialPaddingX, materialPaddingY ); 

	// Left
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, x-w, y, z, w, h, d, materialId, materialPaddingX, materialPaddingY ); 

	// Right
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, x+w, y, z, w, h, d, materialId, materialPaddingX, materialPaddingY ); 

	// Top
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, x, y+h, z, w, h, d, materialId, materialPaddingX, materialPaddingY ); 

	// Bottom
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, x, y-h, z, w, h, d, materialId, materialPaddingX, materialPaddingY ); 
	return returnValue;
}

void GPUKernel::setPrimitiveMaterial( 
	unsigned int index, 
	int   materialId)
{
	LOG_INFO(3,"GPUKernel::setPrimitiveMaterial(" << index << "," << materialId << ")" );
	if( index>=0 && index<=m_primitives.size()) 
	{
		m_primitives[index].materialId = materialId;
		// TODO: updateLight( index );
	}
}

int GPUKernel::getPrimitiveMaterial( unsigned int index )
{
	LOG_INFO(3,"GPUKernel::getPrimitiveMaterial(" << index << ")" );
	unsigned int returnValue(-1);
	if( index>=0 && index<=m_primitives.size()) 
	{
		returnValue = m_primitives[index].materialId;
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
		m_hMaterials[index].textureInfo.x = procedural ? 1 : 0;
		m_hMaterials[index].textureInfo.y = textureId;
		m_hMaterials[index].textureInfo.z = wireframe ? 1 : 0;
		m_hMaterials[index].textureInfo.w = wireframeWidth;
      m_hMaterials[index].fastTransparency.x = fastTransparency ? 1 : 0;
		m_textureTransfered = false;
	}
	else
	{
		LOG_ERROR("GPUKernel::setMaterial: Out of bounds(" << index << "/" << NB_MAX_MATERIALS << ")" );
	}
}

int GPUKernel::getMaterialAttributes( 
	unsigned int index,
	float& r, float& g, float& b, float& noise,
	float& reflection, 
	float& refraction,
	bool&  procedural,
	bool&  wireframe, int& wireframeDepth,
	float& transparency,
	int&   textureId,
	float& specValue, float& specPower, float& specCoef,
   float& innerIllumination, float& illuminationDiffusion, float& illuminationPropagation )
{
	unsigned int returnValue = -1;

	if( index>=0 && index<=m_nbActiveMaterials ) 
	{
		r = m_hMaterials[index].color.x;
		g = m_hMaterials[index].color.y;
		b = m_hMaterials[index].color.z;
		reflection = m_hMaterials[index].reflection.x;
		refraction = m_hMaterials[index].refraction.x;
		procedural   = (m_hMaterials[index].textureInfo.x == 1);
		transparency = m_hMaterials[index].transparency.x;
		textureId = m_hMaterials[index].textureInfo.x;
		specValue = m_hMaterials[index].specular.x;
		specPower = m_hMaterials[index].specular.y;
		specCoef  = m_hMaterials[index].specular.w;
		innerIllumination = m_hMaterials[index].innerIllumination.x;
		illuminationDiffusion = m_hMaterials[index].innerIllumination.y;
		illuminationPropagation = m_hMaterials[index].innerIllumination.z;
		wireframe =  (m_hMaterials[index].textureInfo.z == 1);
		wireframeDepth = m_hMaterials[index].textureInfo.w;
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
   }
   return returnValue;
}


// ---------- Textures ----------
void GPUKernel::setTexture(
	unsigned int index,
	char* texture )
{
	LOG_INFO(3,"GPUKernel::setTexture(" << index << ")" );
	char* idx = m_hTextures+gTextureOffset+index*gTextureSize;
	int j(0);
	for( int i(0); i<gTextureSize; i+=gColorDepth ) {
		idx[j]   = texture[i+2];
		idx[j+1] = texture[i+1];
		idx[j+2] = texture[i];
		j+=gTextureDepth;
	}
}

void GPUKernel::setSceneInfo(
	int    width,
	int    height,
	float  transparentColor,
	bool   shadowsEnabled,
	float  viewDistance,
	float  shadowIntensity,
	int    nbRayIterations,
	float4 backgroundColor,
	bool   supportFor3DVision,
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
	m_sceneInfo.shadowsEnabled.x       = shadowsEnabled;
	m_sceneInfo.viewDistance.x         = viewDistance;
	m_sceneInfo.shadowIntensity.x      = shadowIntensity;
	m_sceneInfo.nbRayIterations.x      = nbRayIterations;
	m_sceneInfo.backgroundColor        = backgroundColor;
	m_sceneInfo.supportFor3DVision.x   = supportFor3DVision;
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
		LOG_INFO(3,"Failed to load GPUKernel " << filename.c_str() );
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
int GPUKernel::addTexture( const std::string& filename )
{
	FILE *filePtr(0); //our file pointer
	BITMAPFILEHEADER bitmapFileHeader; //our bitmap file header
	char *bitmapImage;  //store image data
	BITMAPINFOHEADER bitmapInfoHeader;
	unsigned int imageIdx=0;  //image index counter
	char tempRGB;  //our swap variable

	//open filename in read binary mode
#ifdef WIN32
	fopen_s(&filePtr, filename.c_str(), "rb");
#else
	filePtr = fopen(filename.c_str(), "rb");
#endif
	if (filePtr == NULL) 
   {
      std::cout << "Failed to load " << filename << std::endl;
		return 1;
	}
   std::cout << "Loading " << filename << std::endl;

	//read the bitmap file header
	size_t status = fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

	//verify that this is a bmp file by check bitmap id
	if (bitmapFileHeader.bfType !=0x4D42) {
		fclose(filePtr);
		return 1;
	}

	//read the bitmap info header
	status = fread(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr);

	//move file point to the begging of bitmap data
	fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

	//allocate enough memory for the bitmap image data
	bitmapImage = (char*)malloc(bitmapInfoHeader.biSizeImage);

	//verify memory allocation
	if (!bitmapImage)
	{
		free(bitmapImage);
		fclose(filePtr);
		return 1;
	}

	//read in the bitmap image data
	status = fread( bitmapImage, bitmapInfoHeader.biSizeImage, 1, filePtr);

	//make sure bitmap image data was read
	if (bitmapImage == NULL)
	{
		fclose(filePtr);
		return 0;
	}

	//swap the r and b values to get RGB (bitmap is BGR)
	for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 3)
	{
		tempRGB = bitmapImage[imageIdx];
		bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
		bitmapImage[imageIdx + 2] = tempRGB;
	}

	//close file and return bitmap image data
	fclose(filePtr);

	char* index = m_hTextures + gTextureOffset + (m_nbActiveTextures*bitmapInfoHeader.biSizeImage);
	memcpy( index, bitmapImage, bitmapInfoHeader.biSizeImage );
	m_nbActiveTextures++;

	free( bitmapImage );
	return m_nbActiveTextures-1;
}

#ifdef USE_KINECT
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
				found = true;
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
	return hr;
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

void GPUKernel::saveBitmapToFile( const std::string& filename, char* bitmap, const int width, const int height, const int depth )
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

int GPUKernel::getLight( unsigned int index )
{
   if( index < m_nbActiveLamps )
   {
      return m_hLamps[index];
   }
   return -1;
}

void GPUKernel::setLight( 
	unsigned int index,
	float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2, 
   float w,  float h,  float d, int   materialId, float materialPaddingX, float materialPaddingY )
{
   bool found(false);
   unsigned int i(0);
   while( i<m_nbActiveLamps && !found )
   {
      if( m_hLamps[i] == index )
      {
         found = true;
      }
      else
      {
         ++i;
      }
   }
   if( found )
   {
      setPrimitive( m_hPrimitives[m_hLamps[i]].index.x, x0, y0, z0, x1, y1, z1, x2, y2, z2,
         w,h,d,materialId,materialPaddingX,materialPaddingY);
      m_hPrimitives[m_hLamps[i]].p0.x = x0;
      m_hPrimitives[m_hLamps[i]].p0.y = y0;
      m_hPrimitives[m_hLamps[i]].p0.z = z0;
   }
}
