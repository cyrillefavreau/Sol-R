#ifdef WIN32
#include <windows.h>
#else
#include <string.h>
#include <math.h>
#include <stdlib.h>
#endif

#include <iostream>
#include <vector>

#include "GPUKernel.h"
#include "Logging.h"
#include "Consts.h"

const int MAX_SOURCE_SIZE = 65535;

float4 min4( const float4 a, const float4 b, const float4 c )
{
   float4 r;
   r.x = min(min(a.x,b.x),c.x);
   r.y = min(min(a.y,b.y),c.y);
   r.z = min(min(a.z,b.z),c.z);
   return r;
}

float4 max4( const float4 a, const float4 b, const float4 c )
{
   float4 r;
   r.x = max(max(a.x,b.x),c.x);
   r.y = max(max(a.y,b.y),c.y);
   r.z = max(max(a.z,b.z),c.z);
   return r;
}

// ________________________________________________________________________________
float vectorLength( const float4& vector )
{
	return sqrt( vector.x*vector.x + vector.y*vector.y + vector.z*vector.z );
}

// ________________________________________________________________________________
void normalizeVector( float4& v )
{
	float l = vectorLength( v );
   v.x /= l;
   v.y /= l;
   v.z /= l;
}

float4 crossProduct( const float4& b, const float4& c )
{
	float4 a;
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
	m_hBoxPrimitivesIndex(0), 
	m_hMaterials(0), 
	m_hTextures(0), 
	m_hDepthOfField(0),
	m_hRandoms(0), 
	m_hPrimitivesXYIds(0),
	m_nbActiveBoxes(-1),
	m_nbActivePrimitives(-1), 
	m_nbActiveLamps(-1),
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
}


GPUKernel::~GPUKernel()
{
	LOG_INFO(3,"GPUKernel::~GPUKernel");
	delete m_hPrimitives;
	delete m_hMaterials;
	delete m_hTextures;
	delete m_hBoxPrimitivesIndex;
	delete m_hBoundingBoxes;
	delete m_hLamps;
	delete m_hPrimitivesXYIds;
	delete m_hRandoms;
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
	m_hBoxPrimitivesIndex = new int[NB_MAX_PRIMITIVES];
	memset(m_hBoxPrimitivesIndex,0,NB_MAX_PRIMITIVES*sizeof(int));
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
		m_hRandoms[i] = (rand()%1000-500)/1000.f;
	}
}

/*
________________________________________________________________________________

Sets camera
________________________________________________________________________________
*/
void GPUKernel::setCamera( 
	float4 eye, float4 dir, float4 angles )
{
	LOG_INFO(3,"GPUKernel::setCamera(" << 
		eye.x << "," << eye.y << "," << eye.z << " -> " <<
		dir.x << "," << dir.y << "," << dir.z << " : "  <<
		angles.x << "," << angles.y << "," << angles.z << ")" 
		);
	m_viewPos   = eye;
	m_viewDir   = dir;
	m_angles.x  += angles.x;
	m_angles.y  += angles.y;
	m_angles.z  += angles.z;
	m_angles.w = 0.f;
}

int GPUKernel::addPrimitive( PrimitiveType type )
{
	LOG_INFO(3,"GPUKernel::addPrimitive");
	m_nbActivePrimitives++;
	m_hPrimitives[m_nbActivePrimitives].type.x = type;
	LOG_INFO(3,"m_nbActivePrimitives = " << m_nbActivePrimitives);
	return m_nbActivePrimitives;
}

Primitive* GPUKernel::getPrimitive( const int index )
{
   Primitive* returnValue = NULL;
	if( index>=0 && index<=m_nbActivePrimitives) 
   {
      returnValue = &m_hPrimitives[index];
   }
   return returnValue;
}

void GPUKernel::setPrimitive( 
	int index, int boxId,
	float x0, float y0, float z0, 
	float w,  float h,  float d,
	int   materialId, 
	float materialPaddingX, float materialPaddingY )
{
	setPrimitive( index, boxId, x0, y0, z0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, materialId, materialPaddingX, materialPaddingY );
}

void GPUKernel::setPrimitive( 
	int index, int boxId,
	float x0, float y0, float z0, 
	float x1, float y1, float z1, 
	float w,  float h,  float d,
	int   materialId, 
	float materialPaddingX, float materialPaddingY )
{
	setPrimitive( index, boxId, x0, y0, z0, x1, y1, z1, 0.f, 0.f, 0.f, w, h, d, materialId, materialPaddingX, materialPaddingY );
}

void GPUKernel::setPrimitive( 
	int index, int boxId, 
	float x0, float y0, float z0, 
	float x1, float y1, float z1, 
	float x2, float y2, float z2, 
	float w,  float h,  float d,
	int   materialId, 
	float materialPaddingX, float materialPaddingY )
{
	LOG_INFO(3,"GPUKernel::setPrimitive( " << 
		index << "," << boxId << ",(" << 
		"center (" << x0 << "," << y0 << "," << z0 << ")," <<
		"size (" << w << "," << h << "," << d << ")," <<
		"material (" << materialId << "," << materialPaddingX << "," << materialPaddingY << ")"
		);
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		m_hPrimitives[index].p0.x   = x0;
		m_hPrimitives[index].p0.y   = y0;
		m_hPrimitives[index].p0.z   = z0;
		m_hPrimitives[index].p0.w   = w;
		m_hPrimitives[index].p1.x   = x1;
		m_hPrimitives[index].p1.y   = y1;
		m_hPrimitives[index].p1.z   = z1;
		m_hPrimitives[index].p2.x   = x2;
		m_hPrimitives[index].p2.y   = y2;
		m_hPrimitives[index].p2.z   = z2;
		m_hPrimitives[index].size.x = w;
		m_hPrimitives[index].size.y = h;
		m_hPrimitives[index].size.z = d;
		m_hPrimitives[index].size.w = 0.f; // Not used
		m_hPrimitives[index].materialId.x = materialId;

		switch( m_hPrimitives[index].type.x )
		{
		case ptSphere:
         {
			   m_hPrimitives[index].size.x = w;
			   m_hPrimitives[index].size.y = w;
			   m_hPrimitives[index].size.z = w;
			   m_hPrimitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
			   m_hPrimitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
			   break;
         }
		case ptEllipsoid:
			{
		      m_hPrimitives[index].size.x = w;
		      m_hPrimitives[index].size.y = h;
		      m_hPrimitives[index].size.z = d;
			   m_hPrimitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
			   m_hPrimitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
            break;
         }
		case ptCylinder:
			{
				// Axis
				float4 axis;
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
				m_hPrimitives[index].axis.x = axis.x;
				m_hPrimitives[index].axis.y = axis.y;
				m_hPrimitives[index].axis.z = axis.z;

            m_hPrimitives[index].size.x = (x1 - x0)/2.f;
            m_hPrimitives[index].size.y = (y1 - y0)/2.f;
            m_hPrimitives[index].size.z = (z1 - z0)/2.f;

				// Material
				m_hPrimitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
				m_hPrimitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
				break;
			}
#ifdef USE_KINECT 
		case ptCamera:
			{
				m_hPrimitives[index].n0.x = 0.f;
				m_hPrimitives[index].n0.y = 0.f;
				m_hPrimitives[index].n0.z = 1.f;
				m_hPrimitives[index].materialInfo.x = ( w != 0.f ) ? (gKinectVideoWidth /w/2)*materialPaddingX : 1.f;
				m_hPrimitives[index].materialInfo.y = ( w != 0.f ) ? (gKinectVideoHeight/h/2)*materialPaddingY : 1.f;
				break;
			}
#endif // USE_KINECT
		case ptXYPlane:
			{
				m_hPrimitives[index].n0.x = 0.f;
				m_hPrimitives[index].n0.y = 0.f;
				m_hPrimitives[index].n0.z = 1.f;
				m_hPrimitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
				m_hPrimitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
				break;
			}
		case ptYZPlane:
			{
				m_hPrimitives[index].n0.x = 1.f;
				m_hPrimitives[index].n0.y = 0.f;
				m_hPrimitives[index].n0.z = 0.f;
				m_hPrimitives[index].materialInfo.x = ( d != 0.f ) ? (gTextureWidth /d/2)*materialPaddingX : 1.f;
				m_hPrimitives[index].materialInfo.y = ( h != 0.f ) ? (gTextureHeight/h/2)*materialPaddingY : 1.f;
				break;
			}
		case ptXZPlane:
		case ptCheckboard:
			{
				m_hPrimitives[index].n0.x = 0.f;
				m_hPrimitives[index].n0.y = 1.f;
				m_hPrimitives[index].n0.z = 0.f;
				m_hPrimitives[index].materialInfo.x = ( w != 0.f ) ? (gTextureWidth /w/2)*materialPaddingX : 1.f;
				m_hPrimitives[index].materialInfo.y = ( d != 0.f ) ? (gTextureHeight/d/2)*materialPaddingY : 1.f;
				break;
			}
		case ptTriangle:
			{
            float4 v0,v1;
            v0.x = m_hPrimitives[index].p1.x-m_hPrimitives[index].p0.x;
            v0.y = m_hPrimitives[index].p1.y-m_hPrimitives[index].p0.y;
            v0.z = m_hPrimitives[index].p1.z-m_hPrimitives[index].p0.z;
            normalizeVector(v0);
            
            v1.x = m_hPrimitives[index].p2.x-m_hPrimitives[index].p0.x;
            v1.y = m_hPrimitives[index].p2.y-m_hPrimitives[index].p0.y;
            v1.z = m_hPrimitives[index].p2.z-m_hPrimitives[index].p0.z;
            normalizeVector(v1);

            m_hPrimitives[index].n0 = crossProduct(v0,v1);
            normalizeVector(m_hPrimitives[index].n0);
				break;
			}
		}
		m_hPrimitives[index].n0.w = 1.f; // 0 if other normals differ
		m_hPrimitives[index].n1 = m_hPrimitives[index].n0;
		m_hPrimitives[index].n2 = m_hPrimitives[index].n0;

		updateBoundingBox( boxId, index );
	}
	else
	{
		LOG_ERROR("GPUKernel::setPrimitive: Out of bounds (" << index << "/" << NB_MAX_PRIMITIVES << ")" );
	}
}

void GPUKernel::setPrimitiveNormals( int index, float4 n0, float4 n1, float4 n2 )
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
      normalizeVector(n0);
		m_hPrimitives[index].n0 = n0;
      normalizeVector(n1);
		m_hPrimitives[index].n1 = n1;
      normalizeVector(n2);
		m_hPrimitives[index].n2 = n2;
   }
}

void GPUKernel::resetLamps()
{
	LOG_INFO(3,"GPUKernel::resetLamps()" );
	memset( m_hLamps, 0, sizeof(int)*NB_MAX_LAMPS );
	m_nbActiveLamps = 0;
	int index(0);
	while ( m_nbActiveLamps < NB_MAX_LAMPS && index <= m_nbActivePrimitives )
	{
		/*
		std::cout << m_hPrimitives[index].materialId.x << " (" <<
		m_hMaterials[m_hPrimitives[index].materialId.x].innerIllumination.x << ")" <<
		std::endl;
		*/
		if( m_hMaterials[m_hPrimitives[index].materialId.x].innerIllumination.x != 0.f )
		{
			m_hLamps[m_nbActiveLamps] = index;
			m_nbActiveLamps++;
			LOG_INFO(3,"Lamp added (" << m_nbActiveLamps << "/" << NB_MAX_LAMPS << ")" );
		}
		++index;
	}
	//std::cout << m_nbActiveLamps << " Lamps" << std::endl;
}

int GPUKernel::getPrimitiveAt( int x, int y )
{
	LOG_INFO(3,"GPUKernel::getPrimitiveAt(" << x << "," << y << ")" );
	int returnValue = -1;
	int index = y*m_sceneInfo.width.x+x;
	if( index>=0 && index<m_sceneInfo.width.x*m_sceneInfo.height.x)
		returnValue = m_hPrimitivesXYIds[index];
	return returnValue;
}

void GPUKernel::updateBoundingBox( const int boxId, const int primitiveIndex )
{
	LOG_INFO(3,"GPUKernel::updateBoundingBox(" << boxId << "," << primitiveIndex << ")" );
	if( boxId < NB_MAX_BOXES ) 
	{
		// Bounding Box
		if( boxId > m_nbActiveBoxes ) m_nbActiveBoxes = boxId;

		// Is primitive already in box?
		bool found(false);
		int i(0);
		if( m_hBoundingBoxes[boxId].nbPrimitives.x != 0 )
		{
			while( !found && i<m_hBoundingBoxes[boxId].nbPrimitives.x )
			{
				found = (primitiveIndex == m_hBoxPrimitivesIndex[m_hBoundingBoxes[boxId].startIndex.x+i]);
				i += found ? 0 : 1;
			}
		}

		if( !found  ) 
		{
			// Primitive not in box yet, add it.
			//if( i<NB_MAX_PRIMITIVES_PER_BOX )
			{
				// Shift primitive array to insert index element
				int indexFrom = m_hBoundingBoxes[boxId].startIndex.x+m_hBoundingBoxes[boxId].nbPrimitives.x;
				int indexTo   = indexFrom+1;

				memcpy( &m_hBoxPrimitivesIndex[indexTo], &m_hBoxPrimitivesIndex[indexFrom], NB_MAX_PRIMITIVES-indexTo);
				m_hBoxPrimitivesIndex[indexFrom] = primitiveIndex; // TODO
				m_hBoundingBoxes[boxId].nbPrimitives.x++;

				for( int b(boxId+1); b<NB_MAX_BOXES; ++b) m_hBoundingBoxes[b].startIndex.x++;
			}
			/*
			else
			{
			LOG_INFO(3,"*** ERROR ***: Invalid PrimitiveId: " << m_hBoundingBoxes[boxId].nbPrimitives.x << "/" << NB_MAX_PRIMITIVES_PER_BOX );
			}
			*/
		}
		// Process box size
		float4 corner0;
		float4 corner1;

		switch( m_hPrimitives[primitiveIndex].type.x )
		{
		case ptTriangle: 
		case ptCylinder: 
			{
				corner0 = min4(m_hPrimitives[primitiveIndex].p0,m_hPrimitives[primitiveIndex].p1,m_hPrimitives[primitiveIndex].p2);
				corner1 = max4(m_hPrimitives[primitiveIndex].p0,m_hPrimitives[primitiveIndex].p1,m_hPrimitives[primitiveIndex].p2);
				break;
			}
		default:
			{
				corner0 = m_hPrimitives[primitiveIndex].p0;
				corner1 = m_hPrimitives[primitiveIndex].p0;
				break;
			}
		}

		float4 p0,p1;
		p0.x = ( corner0.x <= corner1.x ) ? corner0.x : corner1.x;
		p0.y = ( corner0.y <= corner1.y ) ? corner0.y : corner1.y;
		p0.z = ( corner0.z <= corner1.z ) ? corner0.z : corner1.z;
		p1.x = ( corner0.x >  corner1.x ) ? corner0.x : corner1.x;
		p1.y = ( corner0.y >  corner1.y ) ? corner0.y : corner1.y;
		p1.z = ( corner0.z >  corner1.z ) ? corner0.z : corner1.z;

		switch( m_hPrimitives[primitiveIndex].type.x )
		{
		case ptCylinder: 
		case ptSphere: 
			{
				p0.x -= m_hPrimitives[primitiveIndex].p0.w;
				p0.y -= m_hPrimitives[primitiveIndex].p0.w;
				p0.z -= m_hPrimitives[primitiveIndex].p0.w;

				p1.x += m_hPrimitives[primitiveIndex].p0.w;
				p1.y += m_hPrimitives[primitiveIndex].p0.w;
				p1.z += m_hPrimitives[primitiveIndex].p0.w;
				break;
			}
		default:
			{
				p0.x -= m_hPrimitives[primitiveIndex].size.x;
				p0.y -= m_hPrimitives[primitiveIndex].size.y;
				p0.z -= m_hPrimitives[primitiveIndex].size.z;
				p1.x += m_hPrimitives[primitiveIndex].size.x;
				p1.y += m_hPrimitives[primitiveIndex].size.y;
				p1.z += m_hPrimitives[primitiveIndex].size.z;
				break;
			}
		}

		if( p0.x < m_hBoundingBoxes[boxId].parameters[0].x ) m_hBoundingBoxes[boxId].parameters[0].x = p0.x;
		if( p0.y < m_hBoundingBoxes[boxId].parameters[0].y ) m_hBoundingBoxes[boxId].parameters[0].y = p0.y;
		if( p0.z < m_hBoundingBoxes[boxId].parameters[0].z ) m_hBoundingBoxes[boxId].parameters[0].z = p0.z;
		if( p1.x > m_hBoundingBoxes[boxId].parameters[1].x ) m_hBoundingBoxes[boxId].parameters[1].x = p1.x;
		if( p1.y > m_hBoundingBoxes[boxId].parameters[1].y ) m_hBoundingBoxes[boxId].parameters[1].y = p1.y;
		if( p1.z > m_hBoundingBoxes[boxId].parameters[1].z ) m_hBoundingBoxes[boxId].parameters[1].z = p1.z;
	}
	else
	{
		LOG_ERROR("*** ERROR ***: Invalid BoxId: " << boxId << "/" << NB_MAX_BOXES );
	}
}

void GPUKernel::resetBox( int boxId, bool resetPrimitives )
{
	LOG_INFO(3,"GPUKernel::resetBox(" << boxId << "," << resetPrimitives << ")" );
	if( resetPrimitives ) 
	{
		m_hBoundingBoxes[boxId].nbPrimitives.x = 0;
	}

	m_hBoundingBoxes[boxId].parameters[0].x = m_sceneInfo.viewDistance.x;
	m_hBoundingBoxes[boxId].parameters[0].y = m_sceneInfo.viewDistance.x;
	m_hBoundingBoxes[boxId].parameters[0].z = m_sceneInfo.viewDistance.x;
	m_hBoundingBoxes[boxId].parameters[1].x = -m_sceneInfo.viewDistance.x;
	m_hBoundingBoxes[boxId].parameters[1].y = -m_sceneInfo.viewDistance.x;
	m_hBoundingBoxes[boxId].parameters[1].z = -m_sceneInfo.viewDistance.x;
}

int GPUKernel::compactBoxes()
{
	LOG_INFO(3,"GPUKernel::compactBoxes" );
	resetLamps();
#if 0
   int i(NB_MAX_BOXES-1);
   bool stop(false);
   while( i>=0 && !stop )
   {
      if(m_hBoundingBoxes[i].nbPrimitives.x != 0)
      {
         stop = true;
      }
      else
      {
         --i;
      }
   }
#else
   std::cout << "Compacting " << m_nbActiveBoxes << " boxes..." << std::endl;
   BoundingBox* tmpBoxes = new BoundingBox[NB_MAX_BOXES];
   int j(11);
   int i(11);
   while( j < NB_MAX_BOXES )
   {
      if( m_hBoundingBoxes[j].nbPrimitives.x != 0 )
      {
         memcpy( &tmpBoxes[i], &m_hBoundingBoxes[j], sizeof(BoundingBox));
         ++i;
      }
      ++j;
   }
   memcpy( &m_hBoundingBoxes[11], &tmpBoxes[11], i*sizeof(BoundingBox));
   delete tmpBoxes;
   std::cout << "Compacted boxes from " << j << " to " << i << std::endl;
#endif // 0
   m_nbActiveBoxes = i;
	return i;
}

void GPUKernel::displayBoxesInfo()
{
	for( int i(0); i<=m_nbActiveBoxes; ++i )
	{
		BoundingBox box = m_hBoundingBoxes[i];
		std::cout << "Box " << i << std::endl;
		std::cout << "- # of primitives: " << box.nbPrimitives.x << std::endl;
		std::cout << "- Corners 1      : " << box.parameters[0].x << "," << box.parameters[0].y << "," << box.parameters[0].z << std::endl;
		std::cout << "- Corners 2      : " << box.parameters[1].x << "," << box.parameters[1].y << "," << box.parameters[1].z << std::endl;
		std::cout << "- Primitives     : " << box.startIndex.x << std::endl;
		for( int j(0); j<box.nbPrimitives.x; ++j )
		{
			int index = m_hBoxPrimitivesIndex[j+box.startIndex.x];
			std::cout << "- - " << 
				index << ":" << 
				"type = " << m_hPrimitives[index].type.x << ", " << 
				"center = (" << m_hPrimitives[index].p0.x << "," << m_hPrimitives[index].p0.y << "," << m_hPrimitives[index].p0.z << "), " << 
				"p1 = (" << m_hPrimitives[index].p1.x << "," << m_hPrimitives[index].p1.y << "," << m_hPrimitives[index].p1.z << ")" <<
				std::endl;
		}
		std::cout << std::endl;
	}
}

void GPUKernel::rotatePrimitives( float4 rotationCenter, float4 angles, int from, int to )
{
	LOG_INFO(3,"GPUKernel::rotatePrimitives(" << from << "->" << to << ")" );
	float4 cosAngles, sinAngles;
	cosAngles.x = cos(angles.x);
	cosAngles.y = cos(angles.y);
	cosAngles.z = cos(angles.z);
	cosAngles.w = 0.f;
	sinAngles.x = sin(angles.x);
	sinAngles.y = sin(angles.y);
	sinAngles.z = sin(angles.z);
	sinAngles.w = 0.f;

	int i=0;
#pragma omp parallel for
	for( i=from; i<=to; ++i )
	{
		resetBox(i, false);
		for( int j=0; j<m_hBoundingBoxes[i].nbPrimitives.x; ++j )
		{
			rotatePrimitive( i, m_hBoxPrimitivesIndex[m_hBoundingBoxes[i].startIndex.x+j], rotationCenter, cosAngles, sinAngles );
		}
	}
}

void GPUKernel::rotateVector( float4& v, const float4 rotationCenter, const float4& cosAngles, const float4& sinAngles )
{
   // Rotate Center
   float4 vector;
   vector.x = v.x - rotationCenter.x;
   vector.y = v.y - rotationCenter.y;
   vector.z = v.z - rotationCenter.z;
   float4 result = vector; 

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
	int boxId, int index, float4 rotationCenter, float4 cosAngles, float4 sinAngles )
{
	LOG_INFO(3,"GPUKernel::rotatePrimitive(" << boxId << "," << index << ")" );
	if( index<=m_nbActivePrimitives ) 
	{
      if( m_hPrimitives[index].type.x == ptTriangle || m_hPrimitives[index].type.x == ptSphere || m_hPrimitives[index].type.x == ptEllipsoid || m_hPrimitives[index].type.x == ptCylinder )
		{
         rotateVector( m_hPrimitives[index].p0, rotationCenter, cosAngles, sinAngles );
			if( m_hPrimitives[index].type.x == ptCylinder || m_hPrimitives[index].type.x == ptTriangle )
			{
            rotateVector( m_hPrimitives[index].p1, rotationCenter, cosAngles, sinAngles );
            rotateVector( m_hPrimitives[index].p2, rotationCenter, cosAngles, sinAngles );
            // Rotate Normals
            rotateVector( m_hPrimitives[index].n0, rotationCenter, cosAngles, sinAngles );
            if( m_hPrimitives[index].type.x == ptTriangle )
            {
               rotateVector( m_hPrimitives[index].n1, rotationCenter, cosAngles, sinAngles );
               rotateVector( m_hPrimitives[index].n2, rotationCenter, cosAngles, sinAngles );
            }
            else
            {
               // Axis
				   float4 axis;
				   axis.x = m_hPrimitives[index].p1.x - m_hPrimitives[index].p0.x;
				   axis.y = m_hPrimitives[index].p1.y - m_hPrimitives[index].p0.y;
				   axis.z = m_hPrimitives[index].p1.z - m_hPrimitives[index].p0.z;
				   float len = sqrtf( axis.x*axis.x + axis.y*axis.y + axis.z*axis.z );
				   axis.x /= len;
				   axis.y /= len;
				   axis.z /= len;
				   m_hPrimitives[index].axis.x = axis.x;
				   m_hPrimitives[index].axis.y = axis.y;
				   m_hPrimitives[index].axis.z = axis.z;
            }
			}
		}
		updateBoundingBox( boxId, index );
	}
	else
	{
		LOG_ERROR( "GPUKernel::rotatePrimitive: Out of bounds(" << index << "/" << m_nbActivePrimitives << ")" );
	}
}

void GPUKernel::translatePrimitives( float x, float y, float z )
{
	LOG_INFO(3,"GPUKernel::translatePrimitives" );
	int i=0;
#pragma omp parallel for
	for( i=0; i<=m_nbActiveBoxes; ++i )
	{
		resetBox(i, false);
		for( int j=0; j<m_hBoundingBoxes[i].nbPrimitives.x; ++j )
		{
			float4 trigoAngles;
			trigoAngles.x = x;
			trigoAngles.y = y;
			trigoAngles.z = z;
			translatePrimitive( m_hBoxPrimitivesIndex[m_hBoundingBoxes[i].startIndex.x+j], i, trigoAngles.x, trigoAngles.y, trigoAngles.z );
		}
	}

}

void GPUKernel::translatePrimitive( 
	int index, int boxId,
	float x, 
	float y, 
	float z )
{
	LOG_INFO(3,"GPUKernel::translatePrimitive(" << boxId << "," << index << ")" );
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		setPrimitive(
			index, boxId,
			m_hPrimitives[index].p0.x+x, m_hPrimitives[index].p0.y+y, m_hPrimitives[index].p0.z+z,
			m_hPrimitives[index].p1.x+x, m_hPrimitives[index].p1.y+y, m_hPrimitives[index].p1.z+x,
			m_hPrimitives[index].p2.x+x, m_hPrimitives[index].p2.y+y, m_hPrimitives[index].p2.z+x,
			m_hPrimitives[index].size.x, m_hPrimitives[index].size.y, m_hPrimitives[index].size.z,
			m_hPrimitives[index].materialId.x, 1, 1 );
		updateBoundingBox( boxId, index );
	}
}

void GPUKernel::getPrimitiveCenter(
	int   index, 
	float& x, 
	float& y, 
	float& z,
	float& w)
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		x = m_hPrimitives[index].p0.x;
		y = m_hPrimitives[index].p0.y;
		z = m_hPrimitives[index].p0.z;
		w = m_hPrimitives[index].p0.w;
	}
}

void GPUKernel::getPrimitiveOtherCenter(
	int   index, 
	float4& p1)
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		p1 = m_hPrimitives[index].p1;
	}
}

void GPUKernel::setPrimitiveCenter(
	int   index, 
	int   boxId,
	float x, 
	float y, 
	float z,
	float w)
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		m_hPrimitives[index].p0.x = x;
		m_hPrimitives[index].p0.y = y;
		m_hPrimitives[index].p0.z = z;
		m_hPrimitives[index].p0.w = w;
		m_hPrimitives[index].size.x = w;
		updateBoundingBox( boxId, index );
	}
}

int GPUKernel::addCube( 
	int boxId,
	float x, float y, float z, 
	float radius, 
	int   martialId, 
	float materialPaddingX, float materialPaddingY )
{
	LOG_INFO(3,"GPUKernel::addCube(" << boxId << ")" );
	return addRectangle(boxId, x,y,z,radius,radius,radius,martialId,materialPaddingX,materialPaddingY);
}

int GPUKernel::addRectangle( 
	int boxId,
	float x, float y, float z, 
	float w, float h, float d,
	int   martialId, 
	float materialPaddingX, float materialPaddingY )
{
	LOG_INFO(3,"GPUKernel::addRectangle(" << boxId << ")" );
	int returnValue;
	// Back
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, boxId, x, y, z+d, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Front
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, boxId, x, y, z-d, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Left
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, boxId, x-w, y, z, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Right
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, boxId, x+w, y, z, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Top
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, boxId, x, y+h, z, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Bottom
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, boxId, x, y-h, z, w, h, d, martialId, materialPaddingX, materialPaddingY ); 
	return returnValue;
}

void GPUKernel::setPrimitiveMaterial( 
	int   index, 
	int   materialId)
{
	LOG_INFO(3,"GPUKernel::setPrimitiveMaterial(" << index << "," << materialId << ")" );
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		m_hPrimitives[index].materialId.x = materialId;
		// TODO: updateLight( index );
	}
}

int GPUKernel::getPrimitiveMaterial( int index )
{
	LOG_INFO(3,"GPUKernel::getPrimitiveMaterial(" << index << ")" );
	int returnValue(-1);
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		returnValue = m_hPrimitives[index].materialId.x;
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

	if( index>=0 && index<=m_nbActiveMaterials ) 
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
		m_hMaterials[index].reflection.x  = reflection;
		m_hMaterials[index].refraction.x  = refraction;
		m_hMaterials[index].transparency.x= transparency;
		m_hMaterials[index].textureInfo.x = procedural ? 1 : 0;
		m_hMaterials[index].textureInfo.y = textureId;
		m_hMaterials[index].textureInfo.z = wireframe ? 1 : 0;
		m_hMaterials[index].textureInfo.w = wireframeWidth;
      m_hMaterials[index].fastTransparency.x = fastTransparency ? 1 : 0;
		m_textureTransfered = false;

		resetLamps();
	}
	else
	{
		LOG_ERROR("GPUKernel::setMaterial: Out of bounds(" << index << "/" << NB_MAX_MATERIALS << ")" );
	}
}

int GPUKernel::getMaterialAttributes( 
	int    index,
	float& r, float& g, float& b, float& noise,
	float& reflection, 
	float& refraction,
	bool&  procedural,
	bool&  wireframe, int& wireframeDepth,
	float& transparency,
	int&   textureId,
	float& specValue, float& specPower, float& specCoef,
   float& innerIllumination )
{
	int returnValue = -1;

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

	if( index>=0 && index<=m_nbActiveMaterials ) 
	{
      returnValue = &m_hMaterials[index];
   }
   return returnValue;
}


// ---------- Textures ----------
void GPUKernel::setTexture(
	int   index,
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
	// Load the kernel source code into the array source_str
	FILE *fp = 0;
	char *source_str = 0;

#ifdef WIN32
	fopen_s( &fp, filename.c_str(), "r");
#else
	fp = fopen( filename.c_str(), "r");
#endif
	if( fp == 0 ) 
	{
		LOG_INFO(3,"Failed to load kernel " << filename.c_str() );
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
	// Load the kernel source code into the array source_str
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
		LOG_INFO(3,"Failed to save kernel " << filename.c_str() );
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
	if (filePtr == NULL) {
		return 1;
	}

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
	int    primitiveIndex, 
	int    boxId,
	float4 skeletonPosition, 
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

bool GPUKernel::getSkeletonPosition( int index, float4& position )
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
