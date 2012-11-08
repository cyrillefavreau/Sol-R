#ifdef WIN32
#include <windows.h>
#endif

#include <iostream>
#include <vector>

#include "GPUKernel.h"
#include "Logging.h"
#include "Consts.h"

const int MAX_SOURCE_SIZE = 65535;

struct CPUInnerBox
{
   int    id;
   float4 parameters[2];
   std::vector<int> primitives;
};

struct CPUOutterBox
{
   float4 parameters[2];
   std::vector<int> innerBoxes;
};

#ifndef WIN32
typedef struct BITMAPFILEHEADER {
  short bfType;
  int bfSize;
  short Reserved1;
  short Reserved2;
  int bfOffBits;
};

typedef struct BITMAPINFOHEADER {
  int biSizeImage;
  long biWidth;
  long biHeight;
};
#endif

GPUKernel::GPUKernel(bool activeLogging, int platform, int device)
 : m_hPrimitives(0), 
   m_hBoxPrimitivesIndex(0), 
   m_hMaterials(0), 
   m_hTextures(0), 
   m_hDepthOfField(0),
   m_hRandoms(0), 
   m_nbActiveInnerBoxes(-1),
   m_nbActiveOutterBoxes(0),
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
	m_texturedTransfered(false)
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
}

void GPUKernel::initBuffers()
{
   LOG_INFO(3,"GPUKernel::initBuffers");
	// Setup World
   m_hPrimitives = new Primitive[NB_MAX_PRIMITIVES];
	memset(m_hPrimitives,0,NB_MAX_PRIMITIVES*sizeof(Primitive) ); 
   m_hMaterials = new Material[NB_MAX_MATERIALS+1];
	memset(m_hMaterials,0,NB_MAX_MATERIALS*sizeof(Material) ); 
   m_hTextures = new char[gTextureWidth*gTextureHeight*gColorDepth*NB_MAX_TEXTURES];
   m_hBoxPrimitivesIndex = new int[NB_MAX_PRIMITIVES];
   memset(m_hBoxPrimitivesIndex,0,NB_MAX_PRIMITIVES*sizeof(int));
   m_hBoundingBoxes = new BoundingBox[NB_MAX_BOXES];
   memset(m_hBoundingBoxes,0,NB_MAX_BOXES*sizeof(BoundingBox));
   m_hLamps = new int[NB_MAX_LAMPS];
   memset(m_hLamps,0,NB_MAX_LAMPS*sizeof(int));

	// Randoms
   int size = m_sceneInfo.width.x*m_sceneInfo.height.x;
	m_hRandoms = new float[size];
	int i;
#pragma omp parallel for
	for( i=0; i<size; ++i)
	{
		m_hRandoms[i] = (rand()%1000-500)/500.f;
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

long GPUKernel::addPrimitive( PrimitiveType type )
{
   LOG_INFO(3,"GPUKernel::addPrimitive");
   m_nbActivePrimitives++;
	m_hPrimitives[m_nbActivePrimitives].type.x = type;
	m_hPrimitives[m_nbActivePrimitives].materialId.x = NO_MATERIAL;
   LOG_INFO(3,"m_nbActivePrimitives = " << m_nbActivePrimitives);
	return m_nbActivePrimitives;
}

void GPUKernel::setPrimitive( 
	int index, int boxId,
	float x0, float y0, float z0, 
	float w,  float h,  float d,
	int   materialId, 
	int   materialPaddingX, int materialPaddingY )
{
	setPrimitive( index, boxId, x0, y0, z0, 0.f, 0.f, 0.f, w, h, d, materialId, materialPaddingX, materialPaddingY );
}

void GPUKernel::setPrimitive( 
	int index, int boxId, 
	float x0, float y0, float z0, 
	float x1, float y1, float z1, 
	float w,  float h,  float d,
   int   materialId, 
	int   materialPaddingX, int materialPaddingY )
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
		m_hPrimitives[index].size.x = w;
		m_hPrimitives[index].size.y = h;
		m_hPrimitives[index].size.z = d;
		m_hPrimitives[index].size.w = 0.f; // Not used
		m_hPrimitives[index].materialId.x = materialId;

		switch( m_hPrimitives[index].type.x )
		{
      case ptSphere:
		   m_hPrimitives[index].size.x = w;
		   m_hPrimitives[index].size.y = w;
		   m_hPrimitives[index].size.z = w;
         m_hPrimitives[index].materialInfo.x = (gTextureWidth /w/2)*materialPaddingX;
         m_hPrimitives[index].materialInfo.y = (gTextureHeight/h/2)*materialPaddingY;
         break;
      case ptCylinder:
		   m_hPrimitives[index].p1.x = 1.f;
		   m_hPrimitives[index].p1.y = 0.5f;
		   m_hPrimitives[index].p1.z = 0.f;
		   m_hPrimitives[index].size.x = w;
		   m_hPrimitives[index].size.y = w;
		   m_hPrimitives[index].size.z = w;
         m_hPrimitives[index].materialInfo.x = (gTextureWidth /w/2)*materialPaddingX;
         m_hPrimitives[index].materialInfo.y = (gTextureHeight/h/2)*materialPaddingY;
         break;
#ifdef USE_KINECT
      case ptCamera:
         {
            m_hPrimitives[index].normal.x = 0.f;
            m_hPrimitives[index].normal.y = 0.f;
            m_hPrimitives[index].normal.z = 1.f;
            m_hPrimitives[index].materialInfo.x = (gKinectVideoWidth /w/2)*materialPaddingX;
            m_hPrimitives[index].materialInfo.y = (gKinectVideoHeight/h/2)*materialPaddingY;
            break;
         }
#endif // USE_KINECT
		case ptXYPlane:
			{
				m_hPrimitives[index].normal.x = 0.f;
				m_hPrimitives[index].normal.y = 0.f;
				m_hPrimitives[index].normal.z = 1.f;
            m_hPrimitives[index].materialInfo.x = (gTextureWidth /w/2)*materialPaddingX;
            m_hPrimitives[index].materialInfo.y = (gTextureHeight/h/2)*materialPaddingY;
				break;
			}
		case ptYZPlane:
			{
				m_hPrimitives[index].normal.x = 1.f;
				m_hPrimitives[index].normal.y = 0.f;
				m_hPrimitives[index].normal.z = 0.f;
            m_hPrimitives[index].materialInfo.x = (gTextureWidth /d/2)*materialPaddingX;
            m_hPrimitives[index].materialInfo.y = (gTextureHeight/h/2)*materialPaddingY;
				break;
			}
		case ptXZPlane:
		case ptCheckboard:
			{
				m_hPrimitives[index].normal.x = 0.f;
				m_hPrimitives[index].normal.y = 1.f;
				m_hPrimitives[index].normal.z = 0.f;
            m_hPrimitives[index].materialInfo.x = (gTextureWidth /w/2)*materialPaddingX;
            m_hPrimitives[index].materialInfo.y = (gTextureHeight/d/2)*materialPaddingY;
				break;
			}
		}


      // Update light
      if( m_nbActiveLamps < NB_MAX_LAMPS )
      {
         if( m_hMaterials[materialId].innerIllumination.x != 0.f )
         {
            bool found(false);
            int i(0);
            while( i<=m_nbActiveLamps && !found )
            {
               if(m_hLamps[i] == index)
                  found = true;
               else
                  ++i;
            }
            if( !found ) 
            {
               m_nbActiveLamps++;
               m_hLamps[m_nbActiveLamps] = index;
               LOG_INFO(3,"Lamp added (" << m_nbActiveLamps+1 << "/" << NB_MAX_LAMPS << ")" );
            }
         }
      }

      updateBoundingBox( boxId, index );
	}
   else
   {
      LOG_ERROR("GPUKernel::setPrimitive: Out of bounds (" << index << "/" << NB_MAX_PRIMITIVES << ")" );
   }
}

void GPUKernel::updateBoundingBox( const int boxId, const int primitiveIndex )
{
   int bid = boxId;
   LOG_INFO(3,"GPUKernel::updateBoundingBox(" << bid << "," << primitiveIndex << ")" );
   if( bid < NB_MAX_BOXES ) 
   {
      // Bounding Box
      //if( boxId> m_nbActiveInnerBoxes ) m_nbActiveInnerBoxes = boxId;

      // Is primitive already in box?
      bool found(false);
      int i(0);
      if( m_hBoundingBoxes[bid].nbPrimitives.x != 0 )
      {
         while( !found && i<m_hBoundingBoxes[bid].nbPrimitives.x )
         {
            found = (primitiveIndex == m_hBoxPrimitivesIndex[m_hBoundingBoxes[bid].startIndex.x+i]);
            i += found ? 0 : 1;
         }
      }

      if( !found  ) 
      {
         // Primitive not in box yet, add it.
         //if( i<NB_MAX_PRIMITIVES_PER_BOX )
         {
            // Shift primitive array to insert index element
            int indexFrom = m_hBoundingBoxes[bid].startIndex.x+m_hBoundingBoxes[bid].nbPrimitives.x;
            int indexTo   = indexFrom+1;

            memcpy( &m_hBoxPrimitivesIndex[indexTo], &m_hBoxPrimitivesIndex[indexFrom], NB_MAX_PRIMITIVES-indexTo);

            m_hBoxPrimitivesIndex[indexFrom] = primitiveIndex; // TODO
            m_hBoundingBoxes[bid].nbPrimitives.x++;
            for( int b(bid+1); b<NB_MAX_BOXES; ++b)
            {
               m_hBoundingBoxes[b].startIndex.x++;
            }
         }
         /*
         else
         {
            LOG_INFO(3,"*** ERROR ***: Invalid PrimitiveId: " << m_hBoundingBoxes[bid].nbPrimitives.x << "/" << NB_MAX_PRIMITIVES_PER_BOX );
         }
         */
      }
      // Process box size
      float w = m_hPrimitives[primitiveIndex].size.x;
      float h = m_hPrimitives[primitiveIndex].size.y;
      float d = m_hPrimitives[primitiveIndex].size.z;
      float x0 = m_hPrimitives[primitiveIndex].p0.x;
      float y0 = m_hPrimitives[primitiveIndex].p0.y;
      float z0 = m_hPrimitives[primitiveIndex].p0.z;

      m_hBoundingBoxes[bid].parameters[0].x = ((x0-w) < m_hBoundingBoxes[bid].parameters[0].x ) ? x0-w : m_hBoundingBoxes[bid].parameters[0].x;
      m_hBoundingBoxes[bid].parameters[0].y = ((y0-h) < m_hBoundingBoxes[bid].parameters[0].y ) ? y0-h : m_hBoundingBoxes[bid].parameters[0].y;
      m_hBoundingBoxes[bid].parameters[0].z = ((z0-d) < m_hBoundingBoxes[bid].parameters[0].z ) ? z0-d : m_hBoundingBoxes[bid].parameters[0].z;
      m_hBoundingBoxes[bid].parameters[1].x = ((x0+w) > m_hBoundingBoxes[bid].parameters[1].x ) ? x0+w : m_hBoundingBoxes[bid].parameters[1].x;
      m_hBoundingBoxes[bid].parameters[1].y = ((y0+h) > m_hBoundingBoxes[bid].parameters[1].y ) ? y0+h : m_hBoundingBoxes[bid].parameters[1].y;
      m_hBoundingBoxes[bid].parameters[1].z = ((z0+d) > m_hBoundingBoxes[bid].parameters[1].z ) ? z0+d : m_hBoundingBoxes[bid].parameters[1].z;
   }
   else
   {
      LOG_ERROR("*** ERROR ***: Invalid BoxId: " << bid << "/" << NB_MAX_BOXES );
   }
}

void GPUKernel::resetBox( int boxId, bool resetPrimitives )
{
   int bid = boxId;
   LOG_INFO(3,"GPUKernel::resetBox(" << bid << "," << resetPrimitives << ")" );
   if( resetPrimitives ) 
   {
      m_hBoundingBoxes[bid].nbPrimitives.x = 0;
   }

   m_hBoundingBoxes[bid].parameters[0].x = m_sceneInfo.viewDistance.x;
   m_hBoundingBoxes[bid].parameters[0].y = m_sceneInfo.viewDistance.x;
   m_hBoundingBoxes[bid].parameters[0].z = m_sceneInfo.viewDistance.x;
   m_hBoundingBoxes[bid].parameters[1].x = -m_sceneInfo.viewDistance.x;
   m_hBoundingBoxes[bid].parameters[1].y = -m_sceneInfo.viewDistance.x;
   m_hBoundingBoxes[bid].parameters[1].z = -m_sceneInfo.viewDistance.x;
}

int GPUKernel::compactBoxes()
{
   LOG_INFO(3,"GPUKernel::compactBoxes" );

   float4 minPos = {  m_sceneInfo.viewDistance.x, m_sceneInfo.viewDistance.x, m_sceneInfo.viewDistance.x, 0.f };
   float4 maxPos = { -m_sceneInfo.viewDistance.x,-m_sceneInfo.viewDistance.x,-m_sceneInfo.viewDistance.x, 0.f };
   for( int i(0); i<m_nbActivePrimitives; ++i)
   {
      minPos.x = (m_hPrimitives[i].p0.x<minPos.x) ? m_hPrimitives[i].p0.x : minPos.x;
      minPos.y = (m_hPrimitives[i].p0.y<minPos.y) ? m_hPrimitives[i].p0.y : minPos.y;
      minPos.z = (m_hPrimitives[i].p0.z<minPos.z) ? m_hPrimitives[i].p0.z : minPos.z;

      maxPos.x = (m_hPrimitives[i].p0.x>maxPos.x) ? m_hPrimitives[i].p0.x : maxPos.x;
      maxPos.y = (m_hPrimitives[i].p0.y>maxPos.y) ? m_hPrimitives[i].p0.y : maxPos.y;
      maxPos.z = (m_hPrimitives[i].p0.z>maxPos.z) ? m_hPrimitives[i].p0.z : maxPos.z;
   }
   float4 size;
   size.x = maxPos.x - minPos.x;
   size.y = maxPos.y - minPos.y;
   size.z = maxPos.z - minPos.z;

   std::vector<CPUInnerBox> innerBoxes;
   int boxSize = 8;
   float4 boxSteps;
   boxSteps.x = size.x*2.f / boxSize;
   boxSteps.y = size.y*2.f / boxSize;
   boxSteps.z = size.z*2.f / boxSize;
   for( int i(0); i<m_nbActivePrimitives; ++i)
   {
      int X = static_cast<int>(( m_hPrimitives[i].p0.x - minPos.x ) / boxSteps.x);
      int Y = static_cast<int>(( m_hPrimitives[i].p0.y - minPos.y ) / boxSteps.y);
      int Z = static_cast<int>(( m_hPrimitives[i].p0.z - minPos.z ) / boxSteps.z);
      int boxId = X*boxSize*boxSize + Y*boxSize + Z;
      bool found(false);
      std::vector<CPUInnerBox>::iterator it = innerBoxes.begin();
      while( !found && it != innerBoxes.end() )
      {
         if( (*it).id == boxId )
            found = true;
         else
            ++it;
      }
      if( !found) 
      {
         CPUInnerBox box;
         //std::cout << "boxId (" << X << "," << Y << "," << Z << ") " << boxId << std::endl;
         box.id = boxId;

         float w = m_hPrimitives[i].size.x;
         float h = m_hPrimitives[i].size.y;
         float d = m_hPrimitives[i].size.z;
         float x0 = m_hPrimitives[i].p0.x;
         float y0 = m_hPrimitives[i].p0.y;
         float z0 = m_hPrimitives[i].p0.z;

         box.parameters[0].x = x0-w;
         box.parameters[0].y = y0-h;
         box.parameters[0].z = z0-d;
         box.parameters[1].x = x0+w;
         box.parameters[1].y = y0+h;
         box.parameters[1].z = z0+d;

         box.primitives.push_back(i);
         innerBoxes.push_back(box);
      }
      else
      {
         CPUInnerBox& box(*it);

         // Process box size
         float w = m_hPrimitives[i].size.x;
         float h = m_hPrimitives[i].size.y;
         float d = m_hPrimitives[i].size.z;
         float x0 = m_hPrimitives[i].p0.x;
         float y0 = m_hPrimitives[i].p0.y;
         float z0 = m_hPrimitives[i].p0.z;

         box.parameters[0].x = ((x0-w) < box.parameters[0].x ) ? x0-w : box.parameters[0].x;
         box.parameters[0].y = ((y0-h) < box.parameters[0].y ) ? y0-h : box.parameters[0].y;
         box.parameters[0].z = ((z0-d) < box.parameters[0].z ) ? z0-d : box.parameters[0].z;
         box.parameters[1].x = ((x0+w) > box.parameters[1].x ) ? x0+w : box.parameters[1].x;
         box.parameters[1].y = ((y0+h) > box.parameters[1].y ) ? y0+h : box.parameters[1].y;
         box.parameters[1].z = ((z0+d) > box.parameters[1].z ) ? z0+d : box.parameters[1].z;
            
         box.primitives.push_back(i);
      }
   }

   // initialize Outter boxes
   std::vector<CPUOutterBox> outterBoxes;
   for(int i(0);i<8;++i)
   {
      CPUOutterBox outterBox;
      outterBox.parameters[0].x =  m_sceneInfo.viewDistance.x;
      outterBox.parameters[0].y =  m_sceneInfo.viewDistance.x;
      outterBox.parameters[0].z =  m_sceneInfo.viewDistance.x;
      outterBox.parameters[1].x = -m_sceneInfo.viewDistance.x;
      outterBox.parameters[1].y = -m_sceneInfo.viewDistance.x;
      outterBox.parameters[1].z = -m_sceneInfo.viewDistance.x;
      outterBoxes.push_back(outterBox);
   }

   // Process innner Boxes
   int ibId=0;
   std::vector<CPUInnerBox>::iterator it = innerBoxes.begin();
   while( it != innerBoxes.end() )
   {
      CPUInnerBox& box(*it);
      if( box.primitives.size() != 0 )
      {
         int index=-1;
         float4 center;
         center.x = (box.parameters[0].x + box.parameters[1].x ) / 2.f;
         center.y = (box.parameters[0].y + box.parameters[1].y ) / 2.f;
         center.z = (box.parameters[0].z + box.parameters[1].z ) / 2.f;

         if( center.x < 0.f )
            if( center.y < 0.f )
               if( center.z < 0.f )
                  index = 0;
               else
                  index = 1;
            else
               if( center.z < 0.f )
                  index = 2;
               else
                  index = 3;
         else 
            if( center.y < 0.f )
               if( center.z < 0.f )
                  index = 4;
               else
                  index = 5;
            else
               if( center.z < 0.f )
                  index = 6;
               else
                  index = 7;
         ///if(index != -1)
         index = 0;
         {
            outterBoxes[index].innerBoxes.push_back(ibId);
            outterBoxes[index].parameters[0].x = (box.parameters[0].x < outterBoxes[index].parameters[0].x) ? box.parameters[0].x : outterBoxes[index].parameters[0].x;
            outterBoxes[index].parameters[0].y = (box.parameters[0].y < outterBoxes[index].parameters[0].y) ? box.parameters[0].y : outterBoxes[index].parameters[0].y;
            outterBoxes[index].parameters[0].z = (box.parameters[0].z < outterBoxes[index].parameters[0].z) ? box.parameters[0].z : outterBoxes[index].parameters[0].z;

            outterBoxes[index].parameters[1].x = (box.parameters[1].x > outterBoxes[index].parameters[1].x) ? box.parameters[1].x : outterBoxes[index].parameters[1].x;
            outterBoxes[index].parameters[1].y = (box.parameters[1].y > outterBoxes[index].parameters[1].y) ? box.parameters[1].y : outterBoxes[index].parameters[1].y;
            outterBoxes[index].parameters[1].z = (box.parameters[1].z > outterBoxes[index].parameters[1].z) ? box.parameters[1].z : outterBoxes[index].parameters[1].z;
            //std::cout << "Inner box: " << ibId << " = " << (*it).primitives.size() << std::endl;
         }
      }
      ibId++;
      it++;
   }

   m_nbActiveOutterBoxes=0;
   m_nbActiveInnerBoxes = innerBoxes.size();

   // --------------------------------------------------------------------------------
   // Reinitialize bounding boxes buffer
   // --------------------------------------------------------------------------------
   memset(m_hBoundingBoxes,0,sizeof(BoundingBox)*NB_MAX_BOXES);
   memset(m_hBoxPrimitivesIndex,0,sizeof(int)*NB_MAX_PRIMITIVES);

   int index(0);
   int startInnerBoxIndex(0);
   std::vector<CPUOutterBox>::iterator itob = outterBoxes.begin();
   while( itob != outterBoxes.end() )
   {
      CPUOutterBox& outterBox(*itob);
      //if( outterBox.innerBoxes.size() != 0 )
      {
         m_hBoundingBoxes[index].nbPrimitives.x = outterBox.innerBoxes.size();
         m_hBoundingBoxes[index].parameters[0]  = outterBox.parameters[0];
         m_hBoundingBoxes[index].parameters[1]  = outterBox.parameters[1];
         m_hBoundingBoxes[index].startIndex.x   = startInnerBoxIndex;

         int iitidx = 0;
         std::vector<int>::const_iterator iit=outterBox.innerBoxes.begin();
         while( iit != outterBox.innerBoxes.end() )
         {
            CPUInnerBox& innerBox = innerBoxes[*iit];
            m_hBoxPrimitivesIndex[startInnerBoxIndex] = (*iit);
            startInnerBoxIndex++;
            ++iit;
         }
         /*
         std::cout << "Outter Box " << index << std::endl;
         std::cout << "  nbPrimitives =" << m_hBoundingBoxes[index].nbPrimitives.x << std::endl;
         std::cout << "  parameters[0]=" << m_hBoundingBoxes[index].parameters[0].x << "," << m_hBoundingBoxes[index].parameters[0].y << "," << m_hBoundingBoxes[index].parameters[0].z << std::endl;
         std::cout << "  parameters[1]=" << m_hBoundingBoxes[index].parameters[1].x << "," << m_hBoundingBoxes[index].parameters[1].y << "," << m_hBoundingBoxes[index].parameters[1].z << std::endl;
         std::cout << "  startIndex   =" << m_hBoundingBoxes[index].startIndex.x << std::endl;
         */
         m_nbActiveOutterBoxes++;
         index++;
      }
      ++itob;
   }

   int startPrimitiveIndex(startInnerBoxIndex);
   it = innerBoxes.begin();
   while( it != innerBoxes.end() )
   {
      CPUInnerBox& innerBox(*it);
      m_hBoundingBoxes[index].nbPrimitives.x = innerBox.primitives.size();
      m_hBoundingBoxes[index].parameters[0]  = innerBox.parameters[0];
      m_hBoundingBoxes[index].parameters[1]  = innerBox.parameters[1];
      m_hBoundingBoxes[index].startIndex.x   = startPrimitiveIndex;

      int iitidx = 0;
      std::vector<int>::const_iterator iit=innerBox.primitives.begin();
      while( iit != innerBox.primitives.end() )
      {
         m_hBoxPrimitivesIndex[startPrimitiveIndex] = (*iit);
         startPrimitiveIndex++;
         ++iit;
      }
      ++index;
      ++it;
   }

   /*
   std::cout << "Outter boxes: " << m_nbActiveOutterBoxes << std::endl;
   std::cout << "Inner boxes : " << m_nbActiveInnerBoxes << std::endl;
   std::cout << "Primitives  : " << m_nbActivePrimitives << std::endl;
   */
   m_nbActiveOutterBoxes = 1; // TODO: Remove!!
   return m_nbActiveInnerBoxes;
}

void GPUKernel::rotatePrimitives( float4 angles, int from, int to )
{
   LOG_INFO(3,"GPUKernel::rotatePrimitives(" << from << "->" << to << ")" );
   float4 trigoAngles;
   trigoAngles.x = cos(angles.x); // cos(x)
   trigoAngles.y = sin(angles.x); // sin(x)
   trigoAngles.z = cos(angles.y); // cos(y)
   trigoAngles.w = sin(angles.y); // sin(y)

#if 0
   int i=0;
   if( to > m_nbActivePrimitives ) to = m_nbActivePrimitives;
#pragma omp parallel for
   for( i=from; i<to; ++i )
   {
      resetBox(i, false);
      for( int j=0; j<m_hBoundingBoxes[m_nbActiveOutterBoxes+i].nbPrimitives.x; ++j )
      {
         rotatePrimitive( i, m_hBoxPrimitivesIndex[m_hBoundingBoxes[m_nbActiveOutterBoxes+i].startIndex.x+j], trigoAngles );
      }
   }
#else
   int i=0;
#pragma omp parallel for
   for( i=0; i<m_nbActivePrimitives; ++i )
   {
      rotatePrimitive( 0, i, trigoAngles );
   }
#endif
   compactBoxes();
}

void GPUKernel::rotatePrimitive( 
	int boxId, int index, float4 angles )
{
   LOG_INFO(3,"GPUKernel::rotatePrimitive(" << boxId << "," << index << ")" );
	if( index>=0 && index<=m_nbActivePrimitives ) 
	{
      if( m_hPrimitives[index].type.x == ptSphere )
      {
         float4 vector = m_hPrimitives[index].p0;
         float4 result = vector; 
         /* X axis */ 
         result.y = vector.y*angles.x - vector.z*angles.y; 
         result.z = vector.y*angles.y + vector.z*angles.x; 
         vector = result; 
         result = vector; 
         /* Y axis */ 
         result.z = vector.z*angles.z - vector.x*angles.w; 
         result.x = vector.z*angles.w + vector.x*angles.z; 
         m_hPrimitives[index].p0 = result; 
      }
#if 0
      updateBoundingBox( boxId, index );
#endif // 0
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
   for( i=0; i<=m_nbActiveInnerBoxes; ++i )
   {
      resetBox(i, false);
      for( int j=0; j<m_hBoundingBoxes[m_nbActiveOutterBoxes+i].nbPrimitives.x; ++j )
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
			m_hPrimitives[index].p1.x+x, m_hPrimitives[index].p1.y+y, m_hPrimitives[index].p1.z+z,
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

long GPUKernel::addCube( 
   int boxId,
	float x, float y, float z, 
	float radius, 
	int   martialId, 
	int   materialPaddingX, int materialPaddingY )
{
   LOG_INFO(3,"GPUKernel::addCube(" << boxId << ")" );
	return addRectangle(boxId, x,y,z,radius,radius,radius,martialId,materialPaddingX,materialPaddingY);
}

long GPUKernel::addRectangle( 
   int boxId,
	float x, float y, float z, 
	float w, float h, float d,
	int   martialId, 
	int   materialPaddingX, int materialPaddingY )
{
   LOG_INFO(3,"GPUKernel::addRectangle(" << boxId << ")" );
	long returnValue;
	// Back
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, boxId, x, y, z+d, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Front
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, boxId, x, y, z-d, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Left
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, boxId, x-w, y, z, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Right
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, boxId, x+w, y, z, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Top
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, boxId, x, y+h, z, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Bottom
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, boxId, x, y-h, z, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 
	return returnValue;
}

void GPUKernel::setPrimitiveMaterial( 
	int   index, 
	int   materialId)
{
   LOG_INFO(3,"GPUKernel::setPrimitiveMaterial(" << index << "," << materialId << ")" );
	if( index>=0 && index<=m_nbActivePrimitives) {
		m_hPrimitives[index].materialId.x = materialId;
	}
}

// ---------- Materials ----------
long GPUKernel::addMaterial()
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
	bool  textured,
	float transparency,
	int   textureId,
	float specValue, float specPower, float innerIllumination, float specCoef )
{
   LOG_INFO(3,"GPUKernel::setMaterial( " << 
      index << "," <<
      "color=(" << r << "," << g << "," << b << ")," <<
      "reflection=" << reflection << "," <<
      "refraction=" << refraction << "," <<
      "transparency=" << transparency << "," <<
      "procedural=" << (textured ? "true" : "false") << "," <<
      "textureId=" << textureId << "," <<
      "specular=(" << specValue << "," << specPower << "," << specCoef << ")," << 
      "innerIllumination=" << innerIllumination 
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
		m_hMaterials[index].textured.x    = textured;
		m_hMaterials[index].textureId.x   = textureId;
      m_texturedTransfered = false;
	}
   else
   {
      LOG_ERROR("GPUKernel::setMaterial: Out of bounds(" << index << "/" << NB_MAX_MATERIALS << ")" );
   }
}

// ---------- Textures ----------
void GPUKernel::setTexture(
	int   index,
	char* texture )
{
   LOG_INFO(3,"GPUKernel::setTexture(" << index << ")" );
	char* idx = m_hTextures+index*gTextureWidth*gTextureHeight*gTextureDepth;
	int j(0);
	for( int i(0); i<gTextureWidth*gTextureHeight*gColorDepth; i += gColorDepth ) {
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
   int    maxPathTracingIterations)
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
long GPUKernel::addTexture( const std::string& filename )
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
	fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

	//verify that this is a bmp file by check bitmap id
	if (bitmapFileHeader.bfType !=0x4D42) {
		fclose(filePtr);
		return 1;
	}

	//read the bitmap info header
	fread(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr);

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
	fread( bitmapImage, bitmapInfoHeader.biSizeImage, 1, filePtr);

	//make sure bitmap image data was read
	if (bitmapImage == NULL)
	{
		fclose(filePtr);
		return NULL;
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

	char* index = m_hTextures + (m_nbActiveTextures*bitmapInfoHeader.biSizeImage);
	memcpy( index, bitmapImage, bitmapInfoHeader.biSizeImage );
	m_nbActiveTextures++;

	free( bitmapImage );
	return m_nbActiveTextures-1;
}

#ifdef USE_KINECT
long GPUKernel::updateSkeletons( 
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
