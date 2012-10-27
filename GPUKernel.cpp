#ifdef WIN32
#include <windows.h>
#endif

#include <iostream>
#include <vector>

#include "GPUKernel.h"
#include "Consts.h"

const int MAX_SOURCE_SIZE = 65535;

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

GPUKernel::GPUKernel(void)
 : m_hPrimitives(0), 
   m_hBoxPrimitivesIndex(0), 
   m_hMaterials(0), 
   m_hTextures(0), 
   m_hDepthOfField(0),
   m_hRandoms(0), 
	m_nbActivePrimitives(-1), 
   m_nbActiveLamps(-1),
   m_nbActiveMaterials(-1),
   m_nbActiveTextures(0),
#if USE_KINECT
   m_hVideo(0), m_hDepth(0), 
	m_skeletons(0), m_hNextDepthFrameEvent(0), m_hNextVideoFrameEvent(0), m_hNextSkeletonEvent(0),
	m_pVideoStreamHandle(0), m_pDepthStreamHandle(0),
	m_skeletonsBody(-1), m_skeletonsLamp(-1), m_skeletonIndex(-1),
#endif // USE_KINECT
	m_texturedTransfered(false)
{
}


GPUKernel::~GPUKernel(void)
{
   delete m_hPrimitives;
   delete m_hMaterials;
   delete m_hTextures;
   delete m_hBoxPrimitivesIndex;
   delete m_hBoundingBoxes;
   delete m_hLamps;
}

void GPUKernel::initBuffers()
{
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
		m_hRandoms[i] = (rand()%1000-500)/50.f;
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
	m_viewPos   = eye;
	m_viewDir   = dir;
	m_angles.x  += angles.x;
	m_angles.y  += angles.y;
	m_angles.z  += angles.z;
 	m_angles.w = 0.f;
}

long GPUKernel::addPrimitive( PrimitiveType type )
{
   m_nbActivePrimitives++;
	m_hPrimitives[m_nbActivePrimitives].type.x = type;
	m_hPrimitives[m_nbActivePrimitives].materialId.x = NO_MATERIAL;
	return m_nbActivePrimitives;
}

void GPUKernel::setPrimitive( 
	int index, int boxId,
	float x0, float y0, float z0, 
	float w,  float h,  float d,
	int   materialId, 
	int   materialPaddingX, int materialPaddingY )
{
	setPrimitive( index, boxId, x0, y0, z0, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, materialId, materialPaddingX, materialPaddingY );
}

void GPUKernel::setPrimitive( 
	int index, int boxId, 
	float x0, float y0, float z0, 
	float x1, float y1, float z1, 
	float x2, float y2, float z2, 
	float w,  float h,  float d,
   int   materialId, 
	int   materialPaddingX, int materialPaddingY )
{
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		m_hPrimitives[index].p0.x   = x0;
		m_hPrimitives[index].p0.y   = y0;
		m_hPrimitives[index].p0.z   = z0;
		m_hPrimitives[index].p0.w   = w;
      m_hPrimitives[index].materialInfo.x = (gTextureWidth /w/2)*materialPaddingX;
      m_hPrimitives[index].materialInfo.y = (gTextureHeight/h/2)*materialPaddingY;
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
				break;
			}
		case ptYZPlane:
			{
				m_hPrimitives[index].normal.x = 1.f;
				m_hPrimitives[index].normal.y = 0.f;
				m_hPrimitives[index].normal.z = 0.f;
				break;
			}
		case ptXZPlane:
		case ptCheckboard:
			{
				m_hPrimitives[index].normal.x = 0.f;
				m_hPrimitives[index].normal.y = 1.f;
				m_hPrimitives[index].normal.z = 0.f;
            m_hPrimitives[index].materialInfo.y = (gTextureHeight/d/2)*materialPaddingY;
				break;
			}
		}


      // Update light
      if( m_nbActiveLamps < 5 /* NB_MAX_LAMPS */ )
      {
         if( m_hMaterials[materialId].innerIllumination.x != 0.f )
         {
            bool found(false);
            int i(0);
            while( i<m_nbActiveLamps && !found )
            {
               if(m_hLamps[i] == index)
                  found = true;
               else
                  ++i;
            }
            if( !found ) 
            {
               std::cout << "Adding lamp" << std::endl;
               m_nbActiveLamps++;
               m_hLamps[m_nbActiveLamps] = index;
            }
         }
      }

      updateBoundingBox( boxId, index );
	}
}

void GPUKernel::updateBoundingBox( const int boxId, const int primitiveIndex )
{
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
            for( int b(boxId+1); b<NB_MAX_BOXES; ++b)
            {
               m_hBoundingBoxes[b].startIndex.x++;
            }
         }
         /*
         else
         {
            std::cout << "*** ERROR ***: Invalid PrimitiveId: " << m_hBoundingBoxes[boxId].nbPrimitives.x << "/" << NB_MAX_PRIMITIVES_PER_BOX << std::endl;
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

      if( (x0 - w) < m_hBoundingBoxes[boxId].parameters[0].x ) m_hBoundingBoxes[boxId].parameters[0].x = x0-w;
      if( (y0 - h) < m_hBoundingBoxes[boxId].parameters[0].y ) m_hBoundingBoxes[boxId].parameters[0].y = y0-h;
      if( (z0 - d) < m_hBoundingBoxes[boxId].parameters[0].z ) m_hBoundingBoxes[boxId].parameters[0].z = z0-d;
      if( (x0 + w) > m_hBoundingBoxes[boxId].parameters[1].x ) m_hBoundingBoxes[boxId].parameters[1].x = x0+w;
      if( (y0 + h) > m_hBoundingBoxes[boxId].parameters[1].y ) m_hBoundingBoxes[boxId].parameters[1].y = y0+h;
      if( (z0 + d) > m_hBoundingBoxes[boxId].parameters[1].z ) m_hBoundingBoxes[boxId].parameters[1].z = z0+d;
   }
   else
   {
      std::cout << "*** ERROR ***: Invalid BoxId: " << boxId << "/" << NB_MAX_BOXES << std::endl;
   }
}

void GPUKernel::resetBox( int boxId, bool resetPrimitives )
{
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
   std::vector<BoundingBox> boxes;
   for( int i(0); i<NB_MAX_BOXES; ++i )
   {
      BoundingBox box = m_hBoundingBoxes[i];
      if( box.nbPrimitives.x != 0 )
      {
         boxes.push_back(box);
      }
   }
   memset(m_hBoundingBoxes,0,sizeof(BoundingBox)*NB_MAX_BOXES);
   int index(0);
   std::vector<BoundingBox>::const_iterator it = boxes.begin();
   while( it != boxes.end() )
   {
      m_hBoundingBoxes[index] = (*it);
      ++index;
      ++it;
   }

   m_nbActiveBoxes = static_cast<int>(boxes.size());
   return m_nbActiveBoxes;
}

void GPUKernel::rotatePrimitives( float4 angles, int from, int to )
{
   float4 trigoAngles;
   trigoAngles.x = cos(angles.x); // cos(x)
   trigoAngles.y = sin(angles.x); // sin(x)
   trigoAngles.z = cos(angles.y); // cos(y)
   trigoAngles.w = sin(angles.y); // sin(y)

   int i=0;
   if( to > m_nbActivePrimitives ) to = m_nbActivePrimitives;
#pragma omp parallel for
   for( i=from; i<to; ++i )
   {
      resetBox(i, false);
      for( int j=0; j<m_hBoundingBoxes[i].nbPrimitives.x; ++j )
      {
         rotatePrimitive( i, m_hBoxPrimitivesIndex[m_hBoundingBoxes[i].startIndex.x+j], trigoAngles );
      }
   }

}

void GPUKernel::rotatePrimitive( 
	int boxId, int primtitiveIndex, float4 angles )
{
	if( primtitiveIndex>=0 && primtitiveIndex<=m_nbActivePrimitives ) 
	{
      if( m_hPrimitives[primtitiveIndex].type.x == ptSphere )
      {
         float4 vector = m_hPrimitives[primtitiveIndex].p0;
         float4 result = vector; 
         /* X axis */ 
         result.y = vector.y*angles.x - vector.z*angles.y; 
         result.z = vector.y*angles.y + vector.z*angles.x; 
         vector = result; 
         result = vector; 
         /* Y axis */ 
         result.z = vector.z*angles.z - vector.x*angles.w; 
         result.x = vector.z*angles.w + vector.x*angles.z; 
         m_hPrimitives[primtitiveIndex].p0 = result; 
      }
      updateBoundingBox( boxId, primtitiveIndex );
	}
}

void GPUKernel::translatePrimitives( float x, float y, float z )
{
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
	if( index>=0 && index<=m_nbActivePrimitives) 
	{
		setPrimitive(
			index, boxId,
			m_hPrimitives[index].p0.x+x, m_hPrimitives[index].p0.y+y, m_hPrimitives[index].p0.z+z,
         0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
         /*
			m_hPrimitives[index].p1.x  , m_hPrimitives[index].p1.y  , m_hPrimitives[index].p1.z,
			m_hPrimitives[index].p2.x  , m_hPrimitives[index].p2.y  , m_hPrimitives[index].p2.z,
         */
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
	return addRectangle(boxId, x,y,z,radius,radius,radius,martialId,materialPaddingX,materialPaddingY);
}

long GPUKernel::addRectangle( 
   int boxId,
	float x, float y, float z, 
	float w, float h, float d,
	int   martialId, 
	int   materialPaddingX, int materialPaddingY )
{
	long returnValue;
	// Back
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, boxId, x, y, z+d, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Front
	returnValue = addPrimitive( ptXYPlane );
	setPrimitive( returnValue, boxId, x, y, z-d, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Left
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, boxId, x-w, y, z, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Right
	returnValue = addPrimitive( ptYZPlane );
	setPrimitive( returnValue, boxId, x+w, y, z, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Top
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, boxId, x, y+h, z, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 

	// Bottom
	returnValue = addPrimitive( ptXZPlane );
	setPrimitive( returnValue, boxId, x, y-h, z, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, w, h, d, martialId, materialPaddingX, materialPaddingY ); 
	return returnValue;
}

void GPUKernel::setPrimitiveMaterial( 
	int   index, 
	int   materialId)
{
	if( index>=0 && index<=m_nbActivePrimitives) {
		m_hPrimitives[index].materialId.x = materialId;
	}
}

// ---------- Materials ----------
long GPUKernel::addMaterial()
{
   m_nbActiveMaterials++;
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
}

// ---------- Textures ----------
void GPUKernel::setTexture(
	int   index,
	char* texture )
{
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
		std::cout << "Failed to load kernel " << filename.c_str() << std::endl;
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
		std::cout << "Failed to save kernel " << filename.c_str() << std::endl;
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

