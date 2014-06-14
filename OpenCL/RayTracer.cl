// Typedefs
typedef float4        Vertex;
typedef int4          PrimitiveXYIdBuffer;
typedef float4        PostProcessingBuffer;
typedef unsigned char BitmapBuffer;
typedef float         RandomBuffer;
typedef int           Lamp;

#undef ADVANCED_GEOMETRY

// Constants
#define NB_MAX_ITERATIONS 20
#define CONST __global
//#define CONST __constant

#define NB_MAX_MATERIALS 65536 // Last 30 materials are reserved
#define gColorDepth 3

#define MATERIAL_NONE -1
#define TEXTURE_NONE -1
#define TEXTURE_MANDELBROT -2
#define TEXTURE_JULIA -3

// Globals
#define PI 3.14159265358979323846f
#define EPSILON 1.f
#define REBOUND_EPSILON 0.00001f

// Kinect
#define KINECT_COLOR_WIDTH  640
#define KINECT_COLOR_HEIGHT 480
#define KINECT_COLOR_DEPTH  4
#define KINECT_COLOR_SIZE   640*480*4

#define KINECT_DEPTH_WIDTH  320
#define KINECT_DEPTH_HEIGHT 240
#define KINECT_DEPTH_DEPTH  2
#define KINECT_DEPTH_SIZE   320*240*2

// 3D vision type
enum VisionType
{
   vtStandard = 0,
   vtAnaglyph = 1,
   vt3DVision = 2,
   vtFishEye  = 3
};

enum OutputType
{
   otOpenGL = 0,
   otDelphi = 1,
   otJPEG   = 2
};

// Scene information
typedef struct
{
   int2   size;                     // Image size
   int    graphicsLevel;            // Graphics level( No Shading=0, Lambert=1, Specular=2, Reflections and Refractions=3, Shadows=4 )
   int    nbRayIterations;          // Maximum number of ray iterations for current frame
   float  transparentColor;         // Value above which r+g+b color is considered as transparent
   float  viewDistance;             // Maximum viewing distance
   float  shadowIntensity;          // Shadow intensity( off=0, pitch black=1)
   float  width3DVision;            // 3D: Distance between both eyes
   float4 backgroundColor;          // Background color
   int    renderingType;            // Rendering type( Standard=0, Anaglyph=1, OculusVR=2, FishEye=3)
   int    renderBoxes;              // Activate bounding box rendering( off=0, on=1 );
   int    pathTracingIteration;     // Current iteration for current frame
   int    maxPathTracingIterations; // Maximum number of iterations for current frame
   int4   misc;                     // x : Bitmap encoding( OpenGL=0, Delphi=1, JPEG=2 )
                                    // y: Timer
                                    // z: Fog( 0: disabled, 1: enabled )
                                    // w: Camera modes( Standard=0, Isometric 3D=1, Antialiazing=2 )
   int4    parameters;              // x: Double-sided triangles( 0:disabled, 1:enabled )
                                    // y: Gradient background( 0:disabled, 1:enabled )
                                    // z: Not used
                                    // w: Not used
   int4    skybox;                  // x: size
                                    // y: material Id
} SceneInfo;

typedef struct
{
   Vertex origin;                    // Origin of the ray
   Vertex direction;                 // Direction of the ray
   Vertex inv_direction;             // Inverted direction( Used for optimal Ray-Box intersection )
   int4   signs;                     // Signs ( Used for optimal Ray-Box intersection )
} Ray;

typedef struct
{
   int2   attribute;                 // ID of the emitting primitive
   Vertex location;                  // Position in space
   float4 color;                     // Light
} LightInformation;

// Enums
enum PrimitiveType 
{
   ptSphere      = 0,
   ptCylinder    = 1,
   ptTriangle    = 2,
   ptCheckboard  = 3,
   ptCamera      = 4,
   ptXYPlane     = 5,
   ptYZPlane     = 6,
   ptXZPlane     = 7,
   ptMagicCarpet = 8,
   ptEnvironment = 9,
   ptEllipsoid   = 10,
   ptQuad        = 11
};

typedef struct
{
   float4 innerIllumination; // x: Inner illumination
   // y: Diffusion strength
   // z: <not used>
   // w: Noise
   float4 color;             // Color( R,G,B )
   float4 specular;          // x: Value
   // y: Power
   // z: <not used>
   // w: <not used>
   float  reflection;        // Reflection rate( No reflection=0 -> Full reflection=1 )
   float  refraction;        // Refraction index( ex: glass=1.33 )
   float  transparency;      // Transparency rate( Opaque=0 -> Full transparency=1 )
   float  opacity;           // Opacity strength
   int4   attributes;        // x: Fast transparency( off=0, on=1 ). Fast transparency produces no shadows 
   //    and drops intersections if rays intersects primitive with the same material ID
   // y: Procedural textures( off=0, on=1 )
   // z: Wireframe( off=0, on=1 ). Wire frame produces no shading
   // w: Wireframe Width
   int4   textureMapping;    // x: U padding
   // y: V padding
   // z: Texture ID (Deprecated)
   // w: Texture color depth
   int4   textureOffset;     // x: Offset in the diffuse map
   // y: Offset in the normal map
   // z: Offset in the bump map
   // w: Offset in the specular map
   int4   textureIds;        // x: Diffuse map
   // y: Normal map
   // z: Bump map
   // w: Specular map
   int4   advancedTextureOffset; // x: Offset in the Reflection map
   // y: Offset in the Transparency map
   // z: not used
   // w: not used
   int4   advancedTextureIds;// x: Reflection map
   // y: Transparency map
   // z: not used
   // w: not used
} Material;

typedef struct
{
   Vertex parameters[2];     // Bottom-Left and Top-Right corners
   int    nbPrimitives;      // Number of primitives in the box
   int    startIndex;        // Index of the first primitive in the box
   int2   indexForNextBox;   // If no intersection, how many of the following boxes can be skipped?
} BoundingBox;

typedef struct
{
   // Vertices
   Vertex p0;
   Vertex p1;
   Vertex p2;
   // Normals
   Vertex n0;
   Vertex n1;
   Vertex n2;
   // Size( x,y,z )
   Vertex size;
   // Type( See PrimitiveType )
   int   type;
   // Index
   int   index;
   // Material ID
   int  materialId;
   // Texture coordinates
   Vertex vt0;
   Vertex vt1;
   Vertex vt2;
} Primitive;

typedef struct
{
   unsigned char* buffer; // Pointer to the texture
   int   offset;          // Offset of the texture in the global texture buffer (the one 
   // that will be transfered to the GPU)
   int3  size;            // Size of the texture
} TextureInformation;

// Post processing effect
enum PostProcessingType 
{
   ppe_none,              // No effect
   ppe_depthOfField,      // Depth of field
   ppe_ambientOcclusion,  // Ambient occlusion
   ppe_radiosity,         // Radiosity
   ppe_oneColor
};

typedef struct
{
   int   type;
   float param1; // pointOfFocus;
   float param2; // strength;
   int   param3; // iterations;
} PostProcessingInfo;

// ________________________________________________________________________________
void saturateVector( float4* v )
{
   (*v).x = ((*v).x<0.f) ? 0.f : (*v).x;
   (*v).y = ((*v).y<0.f) ? 0.f : (*v).y; 
   (*v).z = ((*v).z<0.f) ? 0.f : (*v).z;
   (*v).w = ((*v).w<0.f) ? 0.f : (*v).w;

   (*v).x = ((*v).x>1.f) ? 1.f : (*v).x;
   (*v).y = ((*v).y>1.f) ? 1.f : (*v).y; 
   (*v).z = ((*v).z>1.f) ? 1.f : (*v).z;
   (*v).w = ((*v).w>1.f) ? 1.f : (*v).w;
}

/*
________________________________________________________________________________
incident  : le vecteur normal inverse a la direction d'incidence de la source 
lumineuse
normal    : la normale a l'interface orientee dans le materiau ou se propage le 
rayon incident
reflected : le vecteur normal reflechi
________________________________________________________________________________
*/
#define vectorReflection( __r, __i, __n ) __r = __i-2.f*dot(__i,__n)*__n;

/*
________________________________________________________________________________
incident: le vecteur norm? inverse ? la direction d?incidence de la source 
lumineuse
n1      : index of refraction of original medium
n2      : index of refraction of new medium
________________________________________________________________________________
*/
void vectorRefraction( 
   Vertex*      refracted, 
   const Vertex incident, 
   const float  n1, 
   const Vertex normal, 
   const float  n2 )
{
   (*refracted) = incident;
   if(n1!=n2 && n2!=0.f) 
   {
      float r = n1/n2;
      float cosI = dot( incident, normal );
      float cosT2 = 1.f - r*r*(1.f - cosI*cosI);
      (*refracted) = r*incident + (r*cosI-sqrt( fabs(cosT2) ))*normal;
   }
}

/*
________________________________________________________________________________
__v : Vector to rotate
__c : Center of rotations
__a : Angles
________________________________________________________________________________
*/
void vectorRotation(
   Vertex* vector, 
   const Vertex center,
   const Vertex angles )
{
   Vertex __r = (*vector);
   /* X axis */
   __r.y = (*vector).y*half_cos(angles.x) - (*vector).z*half_sin(angles.x);
   __r.z = (*vector).y*half_sin(angles.x) + (*vector).z*half_cos(angles.x);
   (*vector) = __r;
   __r = (*vector);
   /* Y axis */
   __r.z = (*vector).z*half_cos(angles.y) - (*vector).x*half_sin(angles.y);
   __r.x = (*vector).z*half_sin(angles.y) + (*vector).x*half_cos(angles.y); 
   (*vector) = __r;
}

/*
________________________________________________________________________________

Compute ray attributes
________________________________________________________________________________
*/
void computeRayAttributes(Ray* ray)
{
   (*ray).inv_direction.x = ((*ray).direction.x!=0.f) ? 1.f/(*ray).direction.x : 0.f;
   (*ray).inv_direction.y = ((*ray).direction.y!=0.f) ? 1.f/(*ray).direction.y : 0.f;
   (*ray).inv_direction.z = ((*ray).direction.z!=0.f) ? 1.f/(*ray).direction.z : 0.f;
   (*ray).signs.x = ((*ray).inv_direction.x < 0);
   (*ray).signs.y = ((*ray).inv_direction.y < 0);
   (*ray).signs.z = ((*ray).inv_direction.z < 0);
}

/*
________________________________________________________________________________

Convert float4 into OpenGL RGB color
________________________________________________________________________________
*/
void makeColor(
   const SceneInfo* sceneInfo,
   const float4*    color,
   __global BitmapBuffer*    bitmap,
   int              index)
{
   (*color).x = ((*color).x>1.f) ? 1.f : (*color).x;
   (*color).y = ((*color).y>1.f) ? 1.f : (*color).y; 
   (*color).z = ((*color).z>1.f) ? 1.f : (*color).z;
   (*color).x = ((*color).x<0.f) ? 0.f : (*color).x;
   (*color).y = ((*color).y<0.f) ? 0.f : (*color).y; 
   (*color).z = ((*color).z<0.f) ? 0.f : (*color).z;

   int mdc_index = index*gColorDepth; 
   switch( (*sceneInfo).misc.x )
   {
   case otOpenGL: 
      {
         // OpenGL
         bitmap[mdc_index  ] = (BitmapBuffer)((*color).x*255.f); // Red
         bitmap[mdc_index+1] = (BitmapBuffer)((*color).y*255.f); // Green
         bitmap[mdc_index+2] = (BitmapBuffer)((*color).z*255.f); // Blue
         break;
      }
   case otDelphi: 
      {
         // Delphi
         bitmap[mdc_index  ] = (BitmapBuffer)((*color).z*255.f); // Blue
         bitmap[mdc_index+1] = (BitmapBuffer)((*color).y*255.f); // Green
         bitmap[mdc_index+2] = (BitmapBuffer)((*color).x*255.f); // Red
         break;
      }
   case otJPEG: 
      {
         // JPEG
         bitmap[mdc_index+2] = (BitmapBuffer)((*color).z*255.f); // Blue
         bitmap[mdc_index+1] = (BitmapBuffer)((*color).y*255.f); // Green
         bitmap[mdc_index  ] = (BitmapBuffer)((*color).x*255.f); // Red
         break;
      }
   }
}

/*
________________________________________________________________________________

Mandelbrot Set
________________________________________________________________________________
*/
void juliaSet( 
   CONST Primitive* primitive,
   CONST Material*  materials,
   const SceneInfo* sceneInfo, 
   const float x, 
   const float y, 
   float4* color )
{
   CONST Material* material = &materials[(*primitive).materialId];
   float W = (float)(*material).textureMapping.x;
   float H = (float)(*material).textureMapping.y;

   //pick some values for the constant c, this determines the shape of the Julia Set
   float cRe = -0.7f + 0.4f*sin((*sceneInfo).misc.y/1500.f);
   float cIm = 0.27015f + 0.4f*cos((*sceneInfo).misc.y/2000.f);

   //calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
   float newRe = 1.5f * (x - W / 2.f) / (0.5f * W);
   float newIm = (y - H / 2.f) / (0.5f * H);
   //i will represent the number of iterations
   int n;
   //start the iteration process
   float  maxIterations = 40.f+(*sceneInfo).pathTracingIteration;
   for(n = 0; n<maxIterations; n++)
   {
      //remember value of previous iteration
      float oldRe = newRe;
      float oldIm = newIm;
      //the actual iteration, the real and imaginary part are calculated
      newRe = oldRe * oldRe - oldIm * oldIm + cRe;
      newIm = 2.f * oldRe * oldIm + cIm;
      //if the point is outside the circle with radius 2: stop
      if((newRe * newRe + newIm * newIm) > 4.f) break;
   }
   //use color model conversion to get rainbow palette, make brightness black if maxIterations reached
   //color.x += newRe/4.f;
   //color.z += newIm/4.f;
   (*color).x = 1.f-(*color).x*(n/maxIterations);
   (*color).y = 1.f-(*color).y*(n/maxIterations);
   (*color).z = 1.f-(*color).z*(n/maxIterations);
   (*color).w = 1.f-(n/maxIterations);
}

/*
________________________________________________________________________________

Mandelbrot Set
________________________________________________________________________________
*/
void mandelbrotSet( 
   CONST Primitive* primitive,
   CONST Material*  materials,
   const SceneInfo* sceneInfo, 
   const float x, 
   const float y, 
   float4* color )
{
   CONST Material* material = &materials[(*primitive).materialId];
   float W = (float)(*material).textureMapping.x;
   float H = (float)(*material).textureMapping.y;

   float  MinRe		= -2.f;
   float  MaxRe		=	1.f;
   float  MinIm		= -1.2f;
   float  MaxIm		=	MinIm + (MaxRe - MinRe) * H/W;
   float  Re_factor	=	(MaxRe - MinRe) / (W - 1.f);
   float  Im_factor	=	(MaxIm - MinIm) / (H - 1.f);
   float  maxIterations = NB_MAX_ITERATIONS+(*sceneInfo).pathTracingIteration;

   float c_im = MaxIm - y*Im_factor;
   float c_re = MinRe + x*Re_factor;
   float Z_re = c_re;
   float Z_im = c_im;
   bool isInside = true;
   unsigned n;
   for( n = 0; isInside && n < maxIterations; ++n ) 
   {
      float Z_re2 = Z_re*Z_re;
      float Z_im2 = Z_im*Z_im;
      if ( Z_re2+Z_im2>4.f ) 
      {
         isInside = false;
      }
      Z_im = 2.f*Z_re*Z_im+c_im;
      Z_re = Z_re2 - Z_im2+c_re;
   }

   (*color).x = 1.f-(*color).x*(n/maxIterations);
   (*color).y = 1.f-(*color).y*(n/maxIterations);
   (*color).z = 1.f-(*color).z*(n/maxIterations);
   (*color).w = 1.f-(n/maxIterations);
}


// ----------
// Normal mapping
// --------------------
void normalMap(
   const int   index,
   CONST Material*  material,
   CONST BitmapBuffer*    textures,
   Vertex*     normal,
   const float strength)
{
   int i = (*material).textureOffset.y + index;
   BitmapBuffer r,g;
   r = textures[i  ];
   g = textures[i+1];
   (*normal).x -= strength*(r/256.f-0.5f);
   (*normal).y -= strength*(g/256.f-0.5f);
}

// ----------
// Normal mapping
// --------------------
void bumpMap(
   const int        index,
   CONST Material*  material,
   CONST BitmapBuffer*    textures,
   Vertex*          intersection,
   float*           value)
{
   int i = (*material).textureOffset.z + index;
   BitmapBuffer r,g,b;
   r = textures[i  ];
   g = textures[i+1];
   b = textures[i+2];
   (*value)=10.f*(r+g+b)/768.f;
   //(*intersection).x += d;
   //(*intersection).y += d;
   //(*intersection).z += d;
}

// ----------
// Normal mapping
// --------------------
void specularMap(
   const int        index,
   CONST Material*  material,
   CONST BitmapBuffer*    textures,
   Vertex*          specular)
{
   int i = (*material).textureOffset.w + index;
   BitmapBuffer r,g,b;
   r = textures[i  ];
   g = textures[i+1];
   b = textures[i+2];
   (*specular).x *= (r+g+b)/768.f;
}

// ----------
// Reflection mapping
// --------------------
void reflectionMap(
   const int        index,
   CONST Material*  material,
   CONST BitmapBuffer*    textures,
   Vertex*          attributes)
{
   int i = (*material).advancedTextureOffset.x + index;
   BitmapBuffer r,g,b;
   r = textures[i  ];
   g = textures[i+1];
   b = textures[i+2];
   (*attributes).x *= (r+g+b)/768.f;
}

// ----------
// Transparency mapping
// --------------------
void transparencyMap(
   const int        index,
   CONST Material*  material,
   CONST BitmapBuffer*    textures,
   Vertex*          attributes)
{
   int i = (*material).advancedTextureOffset.y + index;
   BitmapBuffer r,g,b;
   r = textures[i  ];
   g = textures[i+1];
   b = textures[i+2];
   (*attributes).y *= (r+g+b)/768.f;
   (*attributes).z = 10.f*b/256.f;
}

#ifdef ADVANCED_GEOMETRY
/*
________________________________________________________________________________

Sphere texture Mapping
________________________________________________________________________________
*/
float4 sphereUVMapping( 
   CONST Primitive* primitive,
   CONST Material*  materials,
   CONST BitmapBuffer* textures,
   Vertex* intersection,
   Vertex* normal,
   Vertex* specular,
   Vertex* attributes)
{
   CONST Material* material = &materials[(*primitive).materialId];
   float4 result = (*material).color;

   Vertex I=normalize((*intersection)-(*primitive).p0);
   float U = ((atan2(I.x, I.z)/PI)+1.f)*.5f;
   float V = (asin(I.y)/PI)+.5f;

   int u=(*material).textureMapping.x*(U*(*primitive).vt1.x);
   int v=(*material).textureMapping.y*(V*(*primitive).vt1.y);

   if( (*material).textureMapping.x != 0 ) u%=(*material).textureMapping.x;
   if( (*material).textureMapping.y != 0 ) v%=(*material).textureMapping.y;
   if( u>=0 && u<(*material).textureMapping.x && v>=0 && v<(*material).textureMapping.y )
   {
      int A=(v*(*material).textureMapping.x+u)*(*material).textureMapping.w;
      int B=(*material).textureMapping.x*(*material).textureMapping.y*(*material).textureMapping.w;
      int index=A%B;

      // Diffuse
      int i=(*material).textureOffset.x+index;
      BitmapBuffer r,g,b;
      r = textures[i  ];
      g = textures[i+1];
      b = textures[i+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;

      float strength=3.f;
      // Bump mapping
      if( (*material).textureIds.z!=TEXTURE_NONE) bumpMap(index, material, textures, intersection, &strength);
      // Normal mapping
      if( (*material).textureIds.y!=TEXTURE_NONE) normalMap(index, material, textures, normal, strength);
      // Specular mapping
      if( (*material).textureIds.w!=TEXTURE_NONE) specularMap(index, material, textures, specular);
      // Reflection mapping
      if( (*material).advancedTextureIds.x!=TEXTURE_NONE) reflectionMap(index, material, textures, attributes);
      // Transparency mapping
      if( (*material).advancedTextureIds.y!=TEXTURE_NONE) transparencyMap(index, material, textures, attributes);
   }
   return result; 
}

/*
________________________________________________________________________________

Cube texture mapping
________________________________________________________________________________
*/
float4 cubeMapping( 
   const SceneInfo*    sceneInfo,
   CONST Primitive*    primitive, 
   CONST Material*     materials,
   CONST BitmapBuffer* textures,
   Vertex* intersection,
   Vertex* normal,
   Vertex* specular,
   Vertex* attributes)
{
   CONST Material* material = &materials[(*primitive).materialId];
   float4 result = (*material).color;

#ifdef USE_KINECT
   if( primitive.type.x == ptCamera )
   {
      int x = ((*intersection).x-(*primitive).p0.x+(*primitive).size.x)*material.textureMapping.x;
      int y = KINECT_COLOR_HEIGHT - ((*intersection).y-(*primitive).p0.y+(*primitive).size.y)*material.textureMapping.y;

      x = (x+KINECT_COLOR_WIDTH)%KINECT_COLOR_WIDTH;
      y = (y+KINECT_COLOR_HEIGHT)%KINECT_COLOR_HEIGHT;

      if( x>=0 && x<KINECT_COLOR_WIDTH && y>=0 && y<KINECT_COLOR_HEIGHT ) 
      {
         int index = (y*KINECT_COLOR_WIDTH+x)*KINECT_COLOR_DEPTH;
         index = index%(material.textureMapping.x*material.textureMapping.y*material.textureMapping.w);
         BitmapBuffer r = textures[index+2];
         BitmapBuffer g = textures[index+1];
         BitmapBuffer b = textures[index+0];
         result.x = r/256.f;
         result.y = g/256.f;
         result.z = b/256.f;
      }
   }
   else
#endif // USE_KINECT
   {
      int u = (((*primitive).type == ptCheckboard) || ((*primitive).type == ptXZPlane) || ((*primitive).type == ptXYPlane))  ? 
         ((*intersection).x-(*primitive).p0.x+(*primitive).size.x):
      ((*intersection).z-(*primitive).p0.z+(*primitive).size.z);

      int v = (((*primitive).type == ptCheckboard) || ((*primitive).type == ptXZPlane)) ? 
         ((*intersection).z+(*primitive).p0.z+(*primitive).size.z) :
      ((*intersection).y-(*primitive).p0.y+(*primitive).size.y);

      if( (*material).textureMapping.x != 0 ) u = u%(*material).textureMapping.x;
      if( (*material).textureMapping.y != 0 ) v = v%(*material).textureMapping.y;

      if( u>=0 && u<(*material).textureMapping.x && v>=0 && v<(*material).textureMapping.x )
      {
         switch( (*material).textureIds.x )
         {
         case TEXTURE_MANDELBROT: mandelbrotSet( primitive, materials, sceneInfo, u, v, &result ); break;
         case TEXTURE_JULIA: juliaSet( primitive, materials, sceneInfo, u, v, &result ); break;
         default:
            {
               int A = (v*(*material).textureMapping.x+u)*(*material).textureMapping.w;
               int B = (*material).textureMapping.x*(*material).textureMapping.y*(*material).textureMapping.w;
               int index = A%B;
               int i = (*material).textureOffset.x + index;
               BitmapBuffer r,g,b;
               r = textures[i  ];
               g = textures[i+1];
               b = textures[i+2];
               result.x = r/256.f;
               result.y = g/256.f;
               result.z = b/256.f;


               float strength=3.f;
               // Bump mapping
               if( (*material).textureIds.z!=TEXTURE_NONE) bumpMap(index, material, textures, intersection, &strength);
               // Normal mapping
               if( (*material).textureIds.y!=TEXTURE_NONE) normalMap(index, material, textures, normal, strength);
               // Specular mapping
               if( (*material).textureIds.w!=TEXTURE_NONE) specularMap(index, material, textures, specular);
               // Reflection mapping
               if( (*material).advancedTextureIds.x!=TEXTURE_NONE) reflectionMap(index, material, textures, attributes);
               // Transparency mapping
               if( (*material).advancedTextureIds.y!=TEXTURE_NONE) transparencyMap(index, material, textures, attributes);
            }
            break;
         }
      }
   }
   return result;
}
#endif // ADVANCED_GEOMETRY

/*
________________________________________________________________________________

Triangle texture Mapping
________________________________________________________________________________
*/
float4 triangleUVMapping( 
   const SceneInfo* sceneInfo,
   CONST Primitive* primitive,
   CONST Material*        materials,
   CONST BitmapBuffer*    textures,
   Vertex* intersection,
   const Vertex    areas,
   Vertex* normal,
   Vertex* specular,
   Vertex* attributes)
{
   CONST Material* material = &materials[(*primitive).materialId];
   float4 result = (*material).color;

   Vertex T = ((*primitive).vt0*areas.x+(*primitive).vt1*areas.y+(*primitive).vt2*areas.z)/(areas.x+areas.y+areas.z);
   int u = T.x*(*material).textureMapping.x;
   int v = T.y*(*material).textureMapping.y;

   u = u%(*material).textureMapping.x;
   v = v%(*material).textureMapping.y;
   if( u>=0 && u<(*material).textureMapping.x && v>=0 && v<(*material).textureMapping.y )
   {
      switch( (*material).textureIds.x )
      {
      case TEXTURE_MANDELBROT: mandelbrotSet( primitive, materials, sceneInfo, u, v, &result ); break;
      case TEXTURE_JULIA: juliaSet( primitive, materials, sceneInfo, u, v, &result ); break;
      default:
         {
            int A = (v*(*material).textureMapping.x+u)*(*material).textureMapping.w;
            int B = (*material).textureMapping.x*(*material).textureMapping.y*(*material).textureMapping.w;
            int index = A%B;

            // Diffuse
            int i = (*material).textureOffset.x + index;
            BitmapBuffer r,g,b;
            r = textures[i  ];
            g = textures[i+1];
            b = textures[i+2];
#ifdef USE_KINECT
            if( (*material).textureIds.x==0 )
            {
               r = textures[index+2];
               g = textures[index+1];
               b = textures[index  ];
            }
#endif // USE_KINECT
            result.x = r/256.f;
            result.y = g/256.f;
            result.z = b/256.f;

            float strength=3.f;
            // Bump mapping
            if( (*material).textureIds.z!=TEXTURE_NONE) bumpMap(index, material, textures, intersection, &strength);
            // Normal mapping
            if( (*material).textureIds.y!=TEXTURE_NONE) normalMap(index, material, textures, normal, strength);
            // Specular mapping
            if( (*material).textureIds.w!=TEXTURE_NONE) specularMap(index, material, textures, specular);
            // Reflection mapping
            if( (*material).advancedTextureIds.x!=TEXTURE_NONE) reflectionMap(index, material, textures, attributes);
            // Transparency mapping
            if( (*material).advancedTextureIds.y!=TEXTURE_NONE) transparencyMap(index, material, textures, attributes);
         }
      }
   }
   return result; 
}

/*
________________________________________________________________________________

Box intersection
________________________________________________________________________________
*/
bool boxIntersection( 
   CONST BoundingBox* box, 
   const Ray*         ray,
   const float        t0,
   const float        t1)
{
   float tmin, tmax, tymin, tymax, tzmin, tzmax;

   tmin = ((*box).parameters[(*ray).signs.x].x - (*ray).origin.x) * (*ray).inv_direction.x;
   tmax = ((*box).parameters[1-(*ray).signs.x].x - (*ray).origin.x) * (*ray).inv_direction.x;
   tymin = ((*box).parameters[(*ray).signs.y].y - (*ray).origin.y) * (*ray).inv_direction.y;
   tymax = ((*box).parameters[1-(*ray).signs.y].y - (*ray).origin.y) * (*ray).inv_direction.y;

   if ( (tmin > tymax) || (tymin > tmax) )
      return false;

   if (tymin > tmin) tmin = tymin;
   if (tymax < tmax) tmax = tymax;
   tzmin = ((*box).parameters[(*ray).signs.z].z - (*ray).origin.z) * (*ray).inv_direction.z;
   tzmax = ((*box).parameters[1-(*ray).signs.z].z - (*ray).origin.z) * (*ray).inv_direction.z;

   if ( (tmin > tzmax) || (tzmin > tmax) ) 
      return false;

   if (tzmin > tmin) tmin = tzmin;
   if (tzmax < tmax) tmax = tzmax;
   return ( (tmin < t1) && (tmax > t0) );
}

/*
________________________________________________________________________________

Ellipsoid intersection
________________________________________________________________________________
*/
bool ellipsoidIntersection(
   const SceneInfo* sceneInfo,
   CONST Primitive* ellipsoid,
   CONST Material*  materials, 
   const Ray* ray, 
   Vertex* intersection,
   Vertex* normal,
   float* shadowIntensity) 
{
   // Shadow intensity
   (*shadowIntensity) = 1.f;

   // solve the equation sphere-ray to find the intersections
   Vertex O_C = (*ray).origin-(*ellipsoid).p0;
   Vertex dir = normalize((*ray).direction);

   float a = 
      ((dir.x*dir.x)/((*ellipsoid).size.x*(*ellipsoid).size.x))
      + ((dir.y*dir.y)/((*ellipsoid).size.y*(*ellipsoid).size.y))
      + ((dir.z*dir.z)/((*ellipsoid).size.z*(*ellipsoid).size.z));
   float b = 
      ((2.f*O_C.x*dir.x)/((*ellipsoid).size.x*(*ellipsoid).size.x))
      + ((2.f*O_C.y*dir.y)/((*ellipsoid).size.y*(*ellipsoid).size.y))
      + ((2.f*O_C.z*dir.z)/((*ellipsoid).size.z*(*ellipsoid).size.z));
   float c = 
      ((O_C.x*O_C.x)/((*ellipsoid).size.x*(*ellipsoid).size.x))
      + ((O_C.y*O_C.y)/((*ellipsoid).size.y*(*ellipsoid).size.y))
      + ((O_C.z*O_C.z)/((*ellipsoid).size.z*(*ellipsoid).size.z))
      - 1.f;

   float d = ((b*b)-(4.f*a*c));
   if ( d<0.f || a==0.f || b==0.f || c==0.f ) 
   { 
      return false;
   }
   d = sqrt(d); 

   float t1 = (-b+d)/(2.f*a);
   float t2 = (-b-d)/(2.f*a);

   if( t1<=EPSILON && t2<=EPSILON ) return false; // both intersections are behind the ray origin

   float t=0.f;
   if( t1<=EPSILON ) 
      t = t2;
   else 
      if( t2<=EPSILON )
         t = t1;
      else
         t=(t1<t2) ? t1 : t2;

   if( t<EPSILON ) return false; // Too close to intersection
   (*intersection) = (*ray).origin + t*dir;

   (*normal) = (*intersection)-(*ellipsoid).p0;
   (*normal).x = 2.f*(*normal).x/((*ellipsoid).size.x*(*ellipsoid).size.x);
   (*normal).y = 2.f*(*normal).y/((*ellipsoid).size.y*(*ellipsoid).size.y);
   (*normal).z = 2.f*(*normal).z/((*ellipsoid).size.z*(*ellipsoid).size.z);

   (*normal) = normalize(*normal);
   return true;
}

/*
________________________________________________________________________________

Skybox mapping
________________________________________________________________________________
*/
float4 skyboxMapping(
   const SceneInfo*    sceneInfo,
   CONST Material*     materials, 
   CONST BitmapBuffer* textures,
   const Ray*          ray
   ) 
{
   CONST Material* material = &materials[(*sceneInfo).skybox.y];
   float4 result = (*material).color;
   // solve the equation sphere-ray to find the intersections
   Vertex dir = normalize((*ray).direction-(*ray).origin); 

   float a = 2.f*dot(dir,dir);
   float b = 2.f*dot((*ray).origin,dir);
   float c = dot((*ray).origin,(*ray).origin)-((*sceneInfo).skybox.x*(*sceneInfo).skybox.x);
   float d = b*b-2.f*a*c;

   if( d<=0.f || a == 0.f) return result;
   float r = sqrt(d);
   float t1 = (-b-r)/a;
   float t2 = (-b+r)/a;

   if( t1<=EPSILON && t2<=EPSILON ) return result; // both intersections are behind the ray origin

   float t=0.f;
   if( t1<=EPSILON ) 
      t=t2;
   else 
      if( t2<=EPSILON )
         t=t1;
      else
         t=(t1<t2) ? t1 : t2;

   if( t<EPSILON ) return result; // Too close to intersection
   Vertex intersection = normalize((*ray).origin+t*dir);

   // Intersection found, no get skybox color

   float U = ((atan2(intersection.x, intersection.z)/PI)+1.f)*.5f;
   float V = (asin(intersection.y)/PI)+.5f;

   int u=(*material).textureMapping.x*U;
   int v=(*material).textureMapping.y*V;

   if( (*material).textureMapping.x != 0 ) u%=(*material).textureMapping.x;
   if( (*material).textureMapping.y != 0 ) v%=(*material).textureMapping.y;
   if( u>=0 && u<(*material).textureMapping.x && v>=0 && v<(*material).textureMapping.y )
   {
      int A=(v*(*material).textureMapping.x+u)*(*material).textureMapping.w;
      int B=(*material).textureMapping.x*(*material).textureMapping.y*(*material).textureMapping.w;
      int index=A%B;

      // Diffuse
      int i=(*material).textureOffset.x+index;
      BitmapBuffer r,g,b;
      r = textures[i  ];
      g = textures[i+1];
      b = textures[i+2];
      result.x = r/256.f;
      result.y = g/256.f;
      result.z = b/256.f;
   }

   return result;
}

#ifdef ADVANCED_GEOMETRY
/*
________________________________________________________________________________

Sphere intersection
________________________________________________________________________________
*/
bool sphereIntersection(
   const SceneInfo* sceneInfo,
   CONST Primitive* sphere, 
   CONST Material*  materials, 
   const Ray* ray, 
   Vertex*    intersection,
   Vertex*    normal,
   float*     shadowIntensity
   ) 
{
   // solve the equation sphere-ray to find the intersections
   Vertex O_C = (*ray).origin-(*sphere).p0;
   Vertex dir = normalize((*ray).direction); 

   float a = 2.f*dot(dir,dir);
   float b = 2.f*dot(O_C,dir);
   float c = dot(O_C,O_C)-((*sphere).size.x*(*sphere).size.x);
   float d = b*b-2.f*a*c;

   if( d<=0.f || a == 0.f) return false;
   float r = sqrt(d);
   float t1 = (-b-r)/a;
   float t2 = (-b+r)/a;

   if( t1<=EPSILON && t2<=EPSILON ) return false; // both intersections are behind the ray origin

   float t=0.f;
   if( t1<=EPSILON ) 
      t=t2;
   else 
      if( t2<=EPSILON )
         t=t1;
      else
         t=(t1<t2) ? t1 : t2;

   if( t<EPSILON ) return false; // Too close to intersection
   (*intersection) = (*ray).origin+t*dir;

   if( materials[(*sphere).materialId].attributes.y==0) 
   {
      // Compute normal vector
      (*normal) = (*intersection)-(*sphere).p0;
   }
   else
   {
      // Procedural texture
      Vertex newCenter;
      newCenter.x = (*sphere).p0.x + 0.008f*(*sphere).size.x*cos((*sceneInfo).misc.y + (*intersection).x );
      newCenter.y = (*sphere).p0.y + 0.008f*(*sphere).size.y*sin((*sceneInfo).misc.y + (*intersection).y );
      newCenter.z = (*sphere).p0.z + 0.008f*(*sphere).size.z*sin(cos((*sceneInfo).misc.y + (*intersection).z ));
      (*normal) = (*intersection)-newCenter;
   }
   (*normal)=normalize(*normal);

   // Shadow management
   r = dot(dir,(*normal));
   (*shadowIntensity)=(materials[(*sphere).materialId].transparency != 0.f) ? (1.f-fabs(r)) : 1.f;

#if EXTENDED_FEATURES
   // Power textures
   if (materials[(*sphere).materialId].textureInfo.y != TEXTURE_NONE && materials[(*sphere).materialId].transparency != 0 ) 
   {
      Vertex color = sphereUVMapping(sphere, materials, textures, intersection, timer );
      return ((color.x+color.y+color.z) >= (*sceneInfo).transparentColor.x ); 
   }
#endif // 0

   return true;
}

Vertex project( const Vertex A, const Vertex B) 
{
   return B*(dot(A,B)/dot(B,B));
}

/*
________________________________________________________________________________

Cylinder (*intersection)
________________________________________________________________________________
*/
bool cylinderIntersection( 
   const SceneInfo* sceneInfo,
   CONST Primitive* cylinder,
   CONST Material*  materials, 
   const Ray* ray,
   Vertex*    intersection,
   Vertex*    normal,
   float*     shadowIntensity) 
{            
   bool back = false;
   Vertex O_C = (*ray).origin-(*cylinder).p0;
   Vertex dir = (*ray).direction;
   Vertex n   = cross(dir, (*cylinder).n1);

   float ln = length(n);

   // Parallel? (?)
   if((ln<EPSILON)&&(ln>-EPSILON)) return false;

   n = normalize(n);

   float d = fabs(dot(O_C,n));
   if (d>(*cylinder).size.y) return false;

   Vertex O = cross(O_C,(*cylinder).n1);
   float t = -dot(O, n)/ln;
   if( t<0.f ) return false;

   O = normalize(cross(n,(*cylinder).n1));
   float s=fabs( sqrt((*cylinder).size.x*(*cylinder).size.x-d*d) / dot( dir,O ) );

   float t1=t-s;
   float t2=t+s;

   // Calculate intersection point
   (*intersection) = (*ray).origin+t1*dir;
   Vertex HB1 = (*intersection)-(*cylinder).p0;
   Vertex HB2 = (*intersection)-(*cylinder).p1;
   float scale1 = dot(HB1,(*cylinder).n1);
   float scale2 = dot(HB2,(*cylinder).n1);
   // Cylinder length
   if( scale1 < EPSILON || scale2 > EPSILON ) 
   {
      (*intersection) = (*ray).origin+t2*dir;
      back = true;
      HB1 = (*intersection)-(*cylinder).p0;
      HB2 = (*intersection)-(*cylinder).p1;
      scale1 = dot(HB1,(*cylinder).n1);
      scale2 = dot(HB2,(*cylinder).n1);
      // Cylinder length
      if( scale1 < EPSILON || scale2 > EPSILON ) return false;
   }

   Vertex V = (*intersection)-(*cylinder).p2;
   (*normal) = V-project(V,(*cylinder).n1);
   (*normal) = normalize(*normal);
   if(back) (*normal) *= -1.f; 

   // Shadow management
   (*shadowIntensity) = 1.f;
   return true;
}

/*
________________________________________________________________________________

Checkboard (*intersection)
________________________________________________________________________________
*/
bool planeIntersection( 
   const SceneInfo*    sceneInfo,
   CONST Primitive*    primitive,
   CONST Material*     materials,
   CONST BitmapBuffer* textures,
   const Ray*          ray, 
   Vertex*             intersection,
   Vertex*             normal,
   float*              shadowIntensity,
   bool                reverse)
{ 
   bool collision = false;

   float reverted = reverse ? -1.f : 1.f;
   switch( (*primitive).type ) 
   {
   case ptMagicCarpet:
   case ptCheckboard:
      {
         (*intersection).y = (*primitive).p0.y;
         float y = (*ray).origin.y-(*primitive).p0.y;
         if( reverted*(*ray).direction.y<0.f && reverted*(*ray).origin.y>reverted*(*primitive).p0.y) 
         {
            (*normal).x =  0.f;
            (*normal).y =  1.f;
            (*normal).z =  0.f;
            (*intersection).x = (*ray).origin.x+y*(*ray).direction.x/-(*ray).direction.y;
            (*intersection).z = (*ray).origin.z+y*(*ray).direction.z/-(*ray).direction.y;
            collision = 
               fabs((*intersection).x - (*primitive).p0.x) < (*primitive).size.x &&
               fabs((*intersection).z - (*primitive).p0.z) < (*primitive).size.z;
         }
         break;
      }
   case ptXZPlane:
      {
         float y = (*ray).origin.y-(*primitive).p0.y;
         if( reverted*(*ray).direction.y<0.f && reverted*(*ray).origin.y>reverted*(*primitive).p0.y) 
         {
            (*normal).x =  0.f;
            (*normal).y =  1.f;
            (*normal).z =  0.f;
            (*intersection).x = (*ray).origin.x+y*(*ray).direction.x/-(*ray).direction.y;
            (*intersection).y = (*primitive).p0.y;
            (*intersection).z = (*ray).origin.z+y*(*ray).direction.z/-(*ray).direction.y;
            collision = 
               fabs((*intersection).x - (*primitive).p0.x) < (*primitive).size.x &&
               fabs((*intersection).z - (*primitive).p0.z) < (*primitive).size.z;
         }
         if( !collision && reverted*(*ray).direction.y>0.f && reverted*(*ray).origin.y<reverted*(*primitive).p0.y) 
         {
            (*normal).x =  0.f;
            (*normal).y = -1.f;
            (*normal).z =  0.f;
            (*intersection).x = (*ray).origin.x+y*(*ray).direction.x/-(*ray).direction.y;
            (*intersection).y = (*primitive).p0.y;
            (*intersection).z = (*ray).origin.z+y*(*ray).direction.z/-(*ray).direction.y;
            collision = 
               fabs((*intersection).x - (*primitive).p0.x) < (*primitive).size.x &&
               fabs((*intersection).z - (*primitive).p0.z) < (*primitive).size.z;
         }
         break;
      }
   case ptYZPlane:
      {
         float x = (*ray).origin.x-(*primitive).p0.x;
         if( reverted*(*ray).direction.x<0.f && reverted*(*ray).origin.x>reverted*(*primitive).p0.x ) 
         {
            (*normal).x =  1.f;
            (*normal).y =  0.f;
            (*normal).z =  0.f;
            (*intersection).x = (*primitive).p0.x;
            (*intersection).y = (*ray).origin.y+x*(*ray).direction.y/-(*ray).direction.x;
            (*intersection).z = (*ray).origin.z+x*(*ray).direction.z/-(*ray).direction.x;
            collision = 
               fabs((*intersection).y - (*primitive).p0.y) < (*primitive).size.y &&
               fabs((*intersection).z - (*primitive).p0.z) < (*primitive).size.z;
         }
         if( !collision && reverted*(*ray).direction.x>0.f && reverted*(*ray).origin.x<reverted*(*primitive).p0.x ) 
         {
            (*normal).x = -1.f;
            (*normal).y =  0.f;
            (*normal).z =  0.f;
            (*intersection).x = (*primitive).p0.x;
            (*intersection).y = (*ray).origin.y+x*(*ray).direction.y/-(*ray).direction.x;
            (*intersection).z = (*ray).origin.z+x*(*ray).direction.z/-(*ray).direction.x;
            collision = 
               fabs((*intersection).y - (*primitive).p0.y) < (*primitive).size.y &&
               fabs((*intersection).z - (*primitive).p0.z) < (*primitive).size.z;
         }
         break;
      }
   case ptXYPlane:
      {
         float z = (*ray).origin.z-(*primitive).p0.z;
         if( reverted*(*ray).direction.z<0.f && reverted*(*ray).origin.z>reverted*(*primitive).p0.z) 
         {
            (*normal).x =  0.f;
            (*normal).y =  0.f;
            (*normal).z =  1.f;
            (*intersection).z = (*primitive).p0.z;
            (*intersection).x = (*ray).origin.x+z*(*ray).direction.x/-(*ray).direction.z;
            (*intersection).y = (*ray).origin.y+z*(*ray).direction.y/-(*ray).direction.z;
            collision = 
               fabs((*intersection).x - (*primitive).p0.x) < (*primitive).size.x &&
               fabs((*intersection).y - (*primitive).p0.y) < (*primitive).size.y;
         }
         if( !collision && reverted*(*ray).direction.z>0.f && reverted*(*ray).origin.z<reverted*(*primitive).p0.z )
         {
            (*normal).x =  0.f;
            (*normal).y =  0.f;
            (*normal).z = -1.f;
            (*intersection).z = (*primitive).p0.z;
            (*intersection).x = (*ray).origin.x+z*(*ray).direction.x/-(*ray).direction.z;
            (*intersection).y = (*ray).origin.y+z*(*ray).direction.y/-(*ray).direction.z;
            collision = 
               fabs((*intersection).x - (*primitive).p0.x) < (*primitive).size.x &&
               fabs((*intersection).y - (*primitive).p0.y) < (*primitive).size.y;
         }
         break;
      }
   case ptCamera:
      {
         if( reverted*(*ray).direction.z<0.f && reverted*(*ray).origin.z>reverted*(*primitive).p0.z )
         {
            (*normal).x =  0.f;
            (*normal).y =  0.f;
            (*normal).z =  1.f;
            (*intersection).z = (*primitive).p0.z;
            float z = (*ray).origin.z-(*primitive).p0.z;
            (*intersection).x = (*ray).origin.x+z*(*ray).direction.x/-(*ray).direction.z;
            (*intersection).y = (*ray).origin.y+z*(*ray).direction.y/-(*ray).direction.z;
            collision =
               fabs((*intersection).x - (*primitive).p0.x) < (*primitive).size.x &&
               fabs((*intersection).y - (*primitive).p0.y) < (*primitive).size.y;
         }
         break;
      }
   }

   if( collision ) 
   {
      // Shadow intensity
      (*shadowIntensity) = 1.f;

      float4 color = materials[(*primitive).materialId].color;
      if( (*primitive).type == ptCamera || materials[(*primitive).materialId].textureIds.x != TEXTURE_NONE )
      {
         Vertex specular = {0.f,0.f,0.f,0.f}; // TODO?
         Vertex attributes;
         color = cubeMapping(sceneInfo, primitive, materials, textures, intersection, normal, &specular, &attributes );
         (*shadowIntensity) = color.w;
      }

      if( (color.x+color.y+color.z)/3.f >= (*sceneInfo).transparentColor ) 
      {
         collision = false;
      }
   }
   return collision;
}
#endif // ADVANCED_GEOMETRY


/*
________________________________________________________________________________

Triangle intersection
________________________________________________________________________________
*/
bool triangleIntersection( 
   const SceneInfo* sceneInfo,
   CONST Primitive* triangle, 
   const Ray*       ray,
   Vertex*          intersection,
   Vertex*          normal,
   Vertex*          areas,
   float*           shadowIntensity,
   const bool       processingShadows )
{
   // Reject rays using the barycentric coordinates of
   // the intersection point with respect to T.
   Vertex E01=(*triangle).p1-(*triangle).p0;
   Vertex E03=(*triangle).p2-(*triangle).p0;
   Vertex P = cross((*ray).direction,E03);
   float det = dot(E01,P);

   if(fabs(det)<EPSILON) return false;

   Vertex T = (*ray).origin-(*triangle).p0;
   float a = dot(T,P)/det;
   if (a < 0.f || a > 1.f) return false;

   Vertex Q = cross(T,E01);
   float b = dot((*ray).direction,Q)/det;
   if (b < 0.f || b > 1.f) return false;

   // Reject rays using the barycentric coordinates of
   // the intersection point with respect to T'.
   if ((a+b) > 1.f) 
   {
      Vertex E23 = (*triangle).p0-(*triangle).p1;
      Vertex E21 = (*triangle).p1-(*triangle).p1;
      Vertex P_ = cross((*ray).direction,E21);
      float det_ = dot(E23,P_);
      if(fabs(det_) < EPSILON) return false;
      Vertex T_ = (*ray).origin-(*triangle).p2;
      float a_ = dot(T_,P_)/det_;
      if (a_ < 0.f) return false;
      Vertex Q_ = cross(T_,E23);
      float b_ = dot((*ray).direction,Q_)/det_;
      if (b_ < 0.f) return false;
   }

   // Compute the ray parameter of the intersection
   // point.
   float t = dot(E03,Q)/det;
   if (t<0.f) return false;

   // Intersection
   (*intersection) = (*ray).origin + t*(*ray).direction;

   // Normal
   Vertex v0 = ((*triangle).p0 - (*intersection));
   Vertex v1 = ((*triangle).p1 - (*intersection));
   Vertex v2 = ((*triangle).p2 - (*intersection));

   (*areas).x = 0.5f*length(cross( v1,v2 ));
   (*areas).y = 0.5f*length(cross( v0,v2 ));
   (*areas).z = 0.5f*length(cross( v0,v1 ));
   (*areas) = normalize(*areas);

   (*normal) = ((*triangle).n0*(*areas).x + (*triangle).n1*(*areas).y + (*triangle).n2*(*areas).z)/((*areas).x+(*areas).y+(*areas).z);

   if( (*sceneInfo).parameters.x==1 )
   {
      // Double Sided triangles
      // Reject triangles with normal opposite to ray.
      Vertex N=normalize((*ray).direction);
      if( processingShadows )
      {
         if( dot(N,(*normal))<=0.f ) return false;
      }
      else
      {
         if( dot(N,(*normal))>=0.f ) return false;
      }
   }


   Vertex dir = normalize((*ray).direction);
   float r = dot(dir,(*normal));

   if( r>0.f )
   {
      (*normal) *= -1.f;
   }

   // Shadow management
   (*shadowIntensity) = 1.f;
   return true;
}

/*
________________________________________________________________________________

(*intersection) Shader
________________________________________________________________________________
*/
float4 intersectionShader( 
   const SceneInfo* sceneInfo,
   CONST Primitive* primitive, 
   CONST Material*  materials,
   CONST BitmapBuffer* textures,
   Vertex*        intersection,
   const Vertex   areas,
   Vertex*        normal,
   Vertex*        specular,
   Vertex*        attributes)
{
   float4 colorAtIntersection = materials[(*primitive).materialId].color;
   colorAtIntersection.w = 0.f; // w attribute is used to dtermine light intensity of the material

#ifdef ADVANCED_GEOMETRY
   switch( (*primitive).type ) 
   {
   case ptCylinder:
      {
         if(materials[(*primitive).materialId].textureIds.x != TEXTURE_NONE)
         {
            colorAtIntersection = sphereUVMapping(primitive, materials, textures, intersection, normal, specular, attributes );
         }
         break;
      }
   case ptEnvironment:
   case ptSphere:
   case ptEllipsoid:
      {
         if(materials[(*primitive).materialId].textureIds.x != TEXTURE_NONE)
         {
            colorAtIntersection = sphereUVMapping( primitive, materials, textures, intersection, normal, specular, attributes );
         }
         break;
      }
   case ptCheckboard :
      {
         if( materials[(*primitive).materialId].textureIds.x != TEXTURE_NONE ) 
         {
            colorAtIntersection = cubeMapping( sceneInfo, primitive, materials, textures, intersection, normal, specular, attributes );
         }
         else 
         {
            int x = (*sceneInfo).viewDistance + (((*intersection).x - (*primitive).p0.x)/(*primitive).size.x);
            int z = (*sceneInfo).viewDistance + (((*intersection).z - (*primitive).p0.z)/(*primitive).size.x);
            if(x%2==0) 
            {
               if (z%2==0) 
               {
                  colorAtIntersection.x = 1.f-colorAtIntersection.x;
                  colorAtIntersection.y = 1.f-colorAtIntersection.y;
                  colorAtIntersection.z = 1.f-colorAtIntersection.z;
               }
            }
            else 
            {
               if (z%2!=0) 
               {
                  colorAtIntersection.x = 1.f-colorAtIntersection.x;
                  colorAtIntersection.y = 1.f-colorAtIntersection.y;
                  colorAtIntersection.z = 1.f-colorAtIntersection.z;
               }
            }
         }
         break;
      }
   case ptXYPlane:
   case ptYZPlane:
   case ptXZPlane:
   case ptCamera:
      {
         if( materials[(*primitive).materialId].textureIds.x != TEXTURE_NONE ) 
         {
            colorAtIntersection = cubeMapping( sceneInfo, primitive, materials, textures, intersection, normal, specular, attributes);
         }
         break;
      }
   case ptTriangle:
      {
         if( materials[(*primitive).materialId].textureIds.x != TEXTURE_NONE ) 
         {
            colorAtIntersection = triangleUVMapping( sceneInfo, primitive, materials, textures, intersection, areas, normal, specular, attributes );
         }
         break;
      }
   }
#else
   if( materials[(*primitive).materialId].textureIds.x != TEXTURE_NONE ) 
   {
      colorAtIntersection = triangleUVMapping( sceneInfo, primitive, materials, textures, intersection, areas, normal, specular, attributes );
      //printf("1. Reflection=%f\n", (*attributes).x);
   }
#endif // ADVANCED_GEOMETRY
   return colorAtIntersection;
}

/*
________________________________________________________________________________

Shadows computation
We do not consider the object from which the ray is launched...
This object cannot shadow itself !

We now have to find the (*intersection) between the considered object and the ray 
which origin is the considered 3D float4 and which direction is defined by the 
light source center.
@return 1.f when pixel is in the shades
________________________________________________________________________________
*/
float processShadows(
   const SceneInfo* sceneInfo,
   CONST BoundingBox*  boudingBoxes, 
   const int nbActiveBoxes,
   CONST Primitive*    primitives,
   CONST Material*     materials,
   CONST BitmapBuffer* textures,
   const int     nbPrimitives, 
   const Vertex  lampCenter, 
   const Vertex  origin, 
   const int     objectId,
   const int     iteration,
   float4*       color)
{
   float result = 0.f;
   int cptBoxes = 0;
   (*color).x = 0.f;
   (*color).y = 0.f;
   (*color).z = 0.f;
   Ray r;
   r.origin    = origin;
   r.direction = lampCenter-origin;
   computeRayAttributes( &r );
   float minDistance  = (iteration<2) ? (*sceneInfo).viewDistance : (*sceneInfo).viewDistance/(iteration+1);

   while( result<(*sceneInfo).shadowIntensity && cptBoxes<nbActiveBoxes )
   {
      CONST BoundingBox* box = &boudingBoxes[cptBoxes];
      if(boxIntersection(box, &r, 0.f, minDistance))
      {
         int cptPrimitives = 0;
         while( result<(*sceneInfo).shadowIntensity && cptPrimitives<(*box).nbPrimitives)
         {
            Vertex intersection = {0.f,0.f,0.f,0.f};
            Vertex normal       = {0.f,0.f,0.f,0.f};
            Vertex areas        = {0.f,0.f,0.f,0.f};
            float  shadowIntensity = 0.f;

            CONST Primitive* primitive = &primitives[(*box).startIndex+cptPrimitives];
            if( (*primitive).index!=objectId && materials[(*primitive).materialId].attributes.x==0)
            {
               bool hit = false;
#ifdef ADVANCED_GEOMETRY
               switch((*primitive).type)
               {
               case ptSphere   : hit=sphereIntersection   ( sceneInfo, primitive, materials, &r, &intersection, &normal, &shadowIntensity ); break;
               case ptCylinder : hit=cylinderIntersection ( sceneInfo, primitive, materials, &r, &intersection, &normal, &shadowIntensity ); break;
               case ptCamera   : hit=false; break;
               case ptEllipsoid: hit=ellipsoidIntersection( sceneInfo, primitive, materials, &r, &intersection, &normal, &shadowIntensity ); break;
               case ptTriangle : hit=triangleIntersection ( sceneInfo, primitive, &r, &intersection, &normal, &areas, &shadowIntensity, true ); break;
               default         : hit=planeIntersection    ( sceneInfo, primitive, materials, textures, &r, &intersection, &normal, &shadowIntensity, false ); break;
               }
#else
               hit=triangleIntersection( sceneInfo, primitive, &r, &intersection, &normal, &areas, &shadowIntensity, true );
#endif // ADVANCED_GEOMETRY
               if( hit )
               {
                  Vertex O_I = intersection-r.origin;
                  Vertex O_L = r.direction;
                  float l = length(O_I);
                  if( l>EPSILON && l<length(O_L) )
                  {
                     float ratio = shadowIntensity*(*sceneInfo).shadowIntensity;
                     if( materials[(*primitive).materialId].transparency!=0.f )
                     {
                        // Shadow color
                        O_L=normalize(O_L);
                        float a=fabs(dot(O_L,normal));
                        float r = (materials[(*primitive).materialId].transparency==0.f ) ? 1.f : (1.f-0.8f*materials[(*primitive).materialId].transparency);
                        ratio *= r*a;
                        (*color).x  += ratio*(0.3f-0.3f*materials[(*primitive).materialId].color.x);
                        (*color).y  += ratio*(0.3f-0.3f*materials[(*primitive).materialId].color.y);
                        (*color).z  += ratio*(0.3f-0.3f*materials[(*primitive).materialId].color.z);
                     }
                     result += ratio;
                  }
               }
            }
            ++cptPrimitives;
         }
         ++cptBoxes;
      }
      else
      {
         cptBoxes+=(*box).indexForNextBox.x;
      }   
   }
   result = (result>(*sceneInfo).shadowIntensity) ? (*sceneInfo).shadowIntensity : result;
   result = (result<0.f) ? 0.f : result;
   return result;
}

/*
________________________________________________________________________________

Primitive shader
________________________________________________________________________________
*/
float4 primitiveShader(
   const int index,
   const SceneInfo*   sceneInfo,
   const PostProcessingInfo*   postProcessingInfo,
   CONST BoundingBox* boundingBoxes, const int nbActiveBoxes, 
   CONST Primitive* primitives, const int nbActivePrimitives,
   CONST LightInformation* lightInformation, const int lightInformationSize, const int nbActiveLamps,
   CONST Material* materials, CONST BitmapBuffer* textures,
   CONST RandomBuffer* randoms,
   const Vertex origin,
   Vertex* normal, 
   const int    objectId, 
   Vertex*      intersection,
   const Vertex areas,
   float4*      closestColor,
   const int    iteration,
   float4*      refractionFromColor,
   float*       shadowIntensity,
   float4*      totalBlinn,
   Vertex*      attributes)
{
   CONST Primitive* primitive = &(primitives[objectId]);
   CONST Material* material = &materials[(*primitive).materialId];
   float4 lampsColor = { 0.f, 0.f, 0.f, 0.f };

   // Lamp Impact
   (*shadowIntensity) = 0.f;

   // Bump
   Vertex bumpNormal={0.f,0.f,0.f,0.f};

   // Specular
   Vertex specular;
   specular.x=(*material).specular.x;
   specular.y=(*material).specular.y;
   specular.z=(*material).specular.z;

   // Intersection color
   float4 intersectionColor = intersectionShader( sceneInfo, primitive, materials, textures, intersection, areas, &bumpNormal, &specular, attributes );
   (*normal) += bumpNormal;
   (*normal) = normalize((*normal));

   if( (*material).innerIllumination.x!=0.f || (*material).attributes.z!=0 )
   {
      // Wireframe returns constant color
      return intersectionColor; 
   }

   if( (*sceneInfo).graphicsLevel>0 )
   {
      // Final color
#ifdef EXTENDED_FEATURES
      // TODO: Bump effect
      if( materials[(*primitive).materialId].textureIds.x != TEXTURE_NONE)
      {
         (*normal).x = (*normal).x*0.7f+intersectionColor.x*0.3f;
         (*normal).y = (*normal).y*0.7f+intersectionColor.y*0.3f;
         (*normal).z = (*normal).z*0.7f+intersectionColor.z*0.3f;
      }
#endif // EXTENDED_FEATURES

      (*closestColor) *= (*material).innerIllumination.x;
      int C=(lightInformationSize>1) ? 2 : 1;
      for( int c=0; c<C; ++c ) 
      {
         int cptLamp = ((*sceneInfo).pathTracingIteration>=NB_MAX_ITERATIONS) ? ((*sceneInfo).pathTracingIteration%lightInformationSize+C-1) : 0;

         if(lightInformation[cptLamp].attribute.x != (*primitive).index)
         {
            Vertex center;
            // randomize lamp center
            center.x = lightInformation[cptLamp].location.x;
            center.y = lightInformation[cptLamp].location.y;
            center.z = lightInformation[cptLamp].location.z;

            int t = (3*(index+(*sceneInfo).misc.y+(*sceneInfo).pathTracingIteration))%((*sceneInfo).size.x*(*sceneInfo).size.y);
            CONST Material* m=&materials[lightInformation[cptLamp].attribute.y];

            if( (*sceneInfo).pathTracingIteration>=NB_MAX_ITERATIONS && lightInformation[cptLamp].attribute.x>=0 && lightInformation[cptLamp].attribute.x<nbActivePrimitives)
            {
               t = t%((*sceneInfo).size.x*(*sceneInfo).size.y-3);
               float a=10.f*(*sceneInfo).pathTracingIteration/(float)((*sceneInfo).maxPathTracingIterations);
               
               center.x += (*m).innerIllumination.y*randoms[t  ]*a;
               center.y += (*m).innerIllumination.y*randoms[t+1]*a;
               center.z += (*m).innerIllumination.y*randoms[t+2]*a;
            }

            Vertex lightRay = center - (*intersection);
            float lightRayLength=length(lightRay);


            if( lightRayLength<(*m).innerIllumination.z )
            {
               float4 shadowColor = {0.f,0.f,0.f,0.f};
               if( (*sceneInfo).graphicsLevel>3 && 
                  iteration<4 && // No need to process shadows after 4 generations of rays... cannot be seen anyway.
                  (*material).innerIllumination.x==0.f ) 
               {
                  (*shadowIntensity) = processShadows(
                     sceneInfo, boundingBoxes, nbActiveBoxes,
                     primitives, materials, textures, 
                     nbActivePrimitives, center, 
                     (*intersection), lightInformation[cptLamp].attribute.x, iteration, &shadowColor );
               }


               if( (*sceneInfo).graphicsLevel>0 )
               {
                  float photonEnergy = sqrt(lightRayLength/(*m).innerIllumination.z);
                  photonEnergy = (photonEnergy>1.f) ? 1.f : photonEnergy;
                  photonEnergy = (photonEnergy<0.f) ? 0.f : photonEnergy;

                  lightRay = normalize(lightRay);
                  // --------------------------------------------------------------------------------
                  // Lambert
                  // --------------------------------------------------------------------------------
                  float lambert = dot((*normal),lightRay); 
                  // Transparent materials are lighted on both sides but the amount of light received by the dark side
                  // depends on the transparency rate.
                  lambert *= (lambert<0.f) ? -materials[(*primitive).materialId].transparency : 1.f;

                  if( lightInformation[cptLamp].attribute.y != MATERIAL_NONE )
                  {
                     CONST Material* m=&materials[lightInformation[cptLamp].attribute.y];
                     lambert *= (*m).innerIllumination.x; // Lamp illumination
                  }
                  else
                  {
                     lambert *= lightInformation[cptLamp].color.w;
                  }

                  if((*material).innerIllumination.w!=0.f) 
                  {
                     // Randomize lamp intensity depending on material noise, for more realistic rendering
                     lambert *= (1.f+randoms[t]*(*material).innerIllumination.w*100.f); 
                  }
                  lambert *= (1.f-(*shadowIntensity));
                  lambert += (*sceneInfo).backgroundColor.w;
                  lambert *= (1.f-photonEnergy);

                  // Lighted object, not in the shades
                  lampsColor += lambert*lightInformation[cptLamp].color - shadowColor;

                  if( (*sceneInfo).graphicsLevel>1 && (*shadowIntensity)<(*sceneInfo).shadowIntensity )
                  {
                     // --------------------------------------------------------------------------------
                     // Blinn - Phong
                     // --------------------------------------------------------------------------------
                     Vertex viewRay = normalize((*intersection) - origin);
                     Vertex blinnDir = lightRay - viewRay;
                     float temp = sqrt(dot(blinnDir,blinnDir));
                     if (temp != 0.f ) 
                     {
                        // Specular reflection
                        blinnDir = (1.f / temp) * blinnDir;
                        float blinnTerm = dot(blinnDir,(*normal));
                        blinnTerm = ( blinnTerm < 0.f) ? 0.f : blinnTerm;

                        blinnTerm = specular.x*pow(blinnTerm,specular.y);
                        //blinnTerm *= (1.f-(*material).transparency);
                        blinnTerm *= (1.f-photonEnergy);
                        (*totalBlinn).x += lightInformation[cptLamp].color.x*lightInformation[cptLamp].color.w*blinnTerm;
                        (*totalBlinn).y += lightInformation[cptLamp].color.y*lightInformation[cptLamp].color.w*blinnTerm;
                        (*totalBlinn).z += lightInformation[cptLamp].color.z*lightInformation[cptLamp].color.w*blinnTerm;

                        // Get transparency from specular map
                        (*totalBlinn).w = specular.z;
                     }
                  }
               }
            }
         }

         // Light impact on material
         (*closestColor) += intersectionColor*lampsColor;

         // Saturate color
         saturateVector(closestColor);

         (*refractionFromColor) = intersectionColor; // Refraction depending on color;
         saturateVector( totalBlinn );
      }
   }
   return (*closestColor); // TODO
}

/*
________________________________________________________________________________

Intersections with primitives
________________________________________________________________________________
*/
inline bool intersectionWithPrimitives(
   const SceneInfo* sceneInfo,
   CONST BoundingBox* boundingBoxes, 
   const int nbActiveBoxes,
   CONST Primitive* primitives, 
   const int nbActivePrimitives,
   CONST Material* materials, 
   CONST BitmapBuffer* textures,
   const Ray* ray, 
   const int iteration,
   int*    closestPrimitive, 
   Vertex* closestIntersection,
   Vertex* closestNormal,
   Vertex* closestAreas,
   float4* colorBox,
   const int currentMaterialId)
{
   bool intersections = false;
   float minDistance  = (iteration<2) ? (*sceneInfo).viewDistance : (*sceneInfo).viewDistance/(iteration+1);

   Ray r;
   r.origin    = (*ray).origin;
   r.direction = (*ray).direction-(*ray).origin;
   computeRayAttributes( &r );

   Vertex intersection = {0.f,0.f,0.f,0.f};
   Vertex normal       = {0.f,0.f,0.f,0.f};
   bool i = false;
   float shadowIntensity = 0.f;

   int cptBoxes=0;
   while( cptBoxes<nbActiveBoxes )
   {
      CONST BoundingBox* box = &boundingBoxes[cptBoxes];
      if( boxIntersection(box, &r, 0.f, minDistance) )
      {
         // Intersection with Box
         if((*sceneInfo).renderBoxes==0)
         {
            // Intersection with primitive within boxes
            for( int cptPrimitives = 0; cptPrimitives<(*box).nbPrimitives; ++cptPrimitives )
            { 
               CONST Primitive* primitive = &primitives[(*box).startIndex+cptPrimitives];
               CONST Material* material = &materials[(*primitive).materialId];
               if( (*material).attributes.x==0 || ((*material).attributes.x==1 && currentMaterialId != (*primitive).materialId)) // !!!! TEST SHALL BE REMOVED TO INCREASE TRANSPARENCY QUALITY !!!
               {
                  Vertex areas = {0.f,0.f,0.f,0.f};
                  i = false;
#ifdef ADVANCED_GEOMETRY
                  switch( (*primitive).type )
                  {
                  case ptEnvironment :
                  case ptSphere      : i = sphereIntersection( sceneInfo, primitive, materials, &r, &intersection, &normal, &shadowIntensity );  break;
                  case ptCylinder    : i = cylinderIntersection( sceneInfo, primitive, materials, &r, &intersection, &normal, &shadowIntensity); break;
                  case ptEllipsoid   : i = ellipsoidIntersection( sceneInfo, primitive, materials, &r, &intersection, &normal, &shadowIntensity ); break;
                  case ptTriangle    : i = triangleIntersection( sceneInfo, primitive, &r, &intersection, &normal, &areas, &shadowIntensity, false ); break;
                  default            : i = planeIntersection( sceneInfo, primitive, materials, textures, &r, &intersection, &normal, &shadowIntensity, false); break;
                  }
#else
                  i = triangleIntersection( sceneInfo, primitive, &r, &intersection, &normal, &areas, &shadowIntensity, false );
#endif // ADVANCED_GEOMETRY

                  float distance = length(intersection-r.origin);
                  if( i && distance>EPSILON && distance<minDistance ) 
                  {
                     // Only keep intersection with the closest object
                     minDistance            = distance;
                     (*closestPrimitive)    = (*box).startIndex+cptPrimitives;
                     (*closestIntersection) = intersection;
                     (*closestNormal)       = normal;
                     (*closestAreas)        = areas;
                     intersections          = true;
                  }
               }
            }
         }
         else
         {
            (*colorBox)+=materials[(*box).startIndex%NB_MAX_MATERIALS].color/50.f;
         }
         ++cptBoxes;
      }
      else
      {
         cptBoxes += (*box).indexForNextBox.x;
      }
   }
   return intersections;
}

/*
________________________________________________________________________________

Calculate the reflected vector                   
We now have to know the colour of this (*intersection)                                        
Color_from_object will compute the amount of light received by the
(*intersection) float4 and  will also compute the shadows. 
The resulted color is stored in result.                     
The first parameter is the closest object to the (*intersection) (following 
the ray). It can  be considered as a light source if its inner light rate 
is > 0.                            
________________________________________________________________________________
*/
inline float4 launchRay( 
   const int index,
   CONST BoundingBox* boundingBoxes, 
   const int nbActiveBoxes,
   CONST Primitive* primitives, 
   const int nbActivePrimitives,
   CONST LightInformation* lightInformation, 
   const int lightInformationSize, 
   const int nbActiveLamps,
   CONST Material*  materials, 
   CONST BitmapBuffer* textures,
   CONST RandomBuffer* randoms,
   const Ray*       ray, 
   const SceneInfo* sceneInfo,
   const PostProcessingInfo* postProcessingInfo,
   Vertex*          intersection,
   float*           depthOfField,
   CONST PrimitiveXYIdBuffer* primitiveXYId)
{
   float4 intersectionColor   = {0.f,0.f,0.f,0.f};

   Vertex closestIntersection = {0.f,0.f,0.f,0.f};
   Vertex firstIntersection   = {0.f,0.f,0.f,0.f};
   Vertex normal              = {0.f,0.f,0.f,0.f};
   float4 closestColor        = {0.f,0.f,0.f,0.f};
   int    closestPrimitive  = 0;
   bool   carryon           = true;
   Ray    rayOrigin         = (*ray);
   float  initialRefraction = 1.f;
   int    iteration         = 0;
   (*primitiveXYId).x = -1;
   (*primitiveXYId).z = 0;
   int currentMaterialId=-2;

   // TODO
   float  colorContributions[NB_MAX_ITERATIONS];
   float4 colors[NB_MAX_ITERATIONS];
   for( int i=0; i<NB_MAX_ITERATIONS; ++i )
   {
      colorContributions[i] = 0.f;
      colors[i].x = 0.f;
      colors[i].y = 0.f;
      colors[i].z = 0.f;
      colors[i].w = 0.f;
   }

   float4 recursiveBlinn = { 0.f, 0.f, 0.f, 0.f };

   // Variable declarations
   float  shadowIntensity = 0.f;
   float4 refractionFromColor;
   Vertex reflectedTarget;
   float4 colorBox = {0.f,0.f,0.f,0.f};
   Vertex latestIntersection=(*ray).origin;
   float rayLength=0.f;

   // Reflected rays
   int reflectedRays=-1;
   Ray reflectedRay;
   float reflectedRatio;
   bool BRDF=false;

   float4 rBlinn = {0.f,0.f,0.f,0.f};
   int currentMaxIteration = ( (*sceneInfo).graphicsLevel<3 ) ? 1 : (*sceneInfo).nbRayIterations+(*sceneInfo).pathTracingIteration;
   currentMaxIteration = (currentMaxIteration>NB_MAX_ITERATIONS) ? NB_MAX_ITERATIONS : currentMaxIteration;

   while( iteration<currentMaxIteration && rayLength<(*sceneInfo).viewDistance && carryon ) 
   {
      Vertex areas = {0.f,0.f,0.f,0.f};
      // If no intersection with lamps detected. Now compute intersection with Primitives
      if( carryon ) 
      {
         carryon = intersectionWithPrimitives(
            sceneInfo,
            boundingBoxes, nbActiveBoxes,
            primitives, nbActivePrimitives,
            materials, textures,
            &rayOrigin,
            iteration,  
            &closestPrimitive, &closestIntersection, 
            &normal, &areas, &colorBox, currentMaterialId);
      }

      if( carryon ) 
      {
         currentMaterialId = primitives[closestPrimitive].materialId;

         if ( iteration==0 )
         {
            colors[iteration].x = 0.f;
            colors[iteration].y = 0.f;
            colors[iteration].z = 0.f;
            colors[iteration].w = 0.f;
            colorContributions[iteration]=1.f;

            firstIntersection=closestIntersection;
            latestIntersection=closestIntersection;

            // Primitive ID for current pixel
            (*primitiveXYId).x = primitives[closestPrimitive].index;

         }

         Vertex attributes;
         attributes.x=materials[primitives[closestPrimitive].materialId].reflection;
         attributes.y=materials[primitives[closestPrimitive].materialId].transparency;
         attributes.z=materials[primitives[closestPrimitive].materialId].refraction;
         attributes.w=materials[primitives[closestPrimitive].materialId].opacity;

         // Get object color
         rBlinn.w = attributes.y;
         colors[iteration] = primitiveShader(
            index, 
            sceneInfo, postProcessingInfo,
            boundingBoxes, nbActiveBoxes, 
            primitives, nbActivePrimitives, 
            lightInformation, lightInformationSize, nbActiveLamps,
            materials, textures, 
            randoms, rayOrigin.origin, &normal, 
            closestPrimitive, &closestIntersection, areas, &closestColor,
            iteration, &refractionFromColor, &shadowIntensity, &rBlinn, &attributes );

         // Primitive illumination
         float colorLight=colors[iteration].x+colors[iteration].y+colors[iteration].z;
         (*primitiveXYId).z += (colorLight>(*sceneInfo).transparentColor) ? 16 : 0;

         float segmentLength=length(closestIntersection-latestIntersection);
         latestIntersection=closestIntersection;
         // ----------
         // Refraction
         // ----------
         float transparency=attributes.y;
         float a=0.f;
         if(transparency!=0.f) // Transparency
         {
            float refraction = attributes.z;

            // Back of the object? If so, reset refraction to 1.f (air)
            if(initialRefraction==refraction)
            {
               // Opacity
               refraction = 1.f;
               float length=segmentLength*(attributes.w*(1.f-transparency));
               rayLength+=length;
               rayLength=(rayLength>(*sceneInfo).viewDistance) ? (*sceneInfo).viewDistance : rayLength;
               a=(rayLength/(*sceneInfo).viewDistance);
               colors[iteration].x-=a;
               colors[iteration].y-=a;
               colors[iteration].z-=a;
            }

            // Actual refraction
            Vertex O_E = normalize(rayOrigin.origin - closestIntersection);
            vectorRefraction( &rayOrigin.direction, O_E, refraction, normal, initialRefraction );
            reflectedTarget = closestIntersection-rayOrigin.direction;

            colorContributions[iteration]=transparency-a;

            // Prepare next ray
            initialRefraction=refraction;

            if( reflectedRays==-1 && attributes.x!=0.f ) // Reflection
            {
               vectorReflection( reflectedRay.direction, O_E, normal );
               Vertex rt = closestIntersection - reflectedRay.direction;

               reflectedRay.origin = closestIntersection+rt*REBOUND_EPSILON;
               reflectedRay.direction = rt;
               reflectedRatio = attributes.x;
               reflectedRays=iteration;
            }
         }
         else
         {
            rayLength+=segmentLength;
            if(attributes.x!=0.f) // Reflection
            {
               Vertex O_E = rayOrigin.origin - closestIntersection;
               vectorReflection( rayOrigin.direction, O_E, normal );
               reflectedTarget = closestIntersection - rayOrigin.direction;
               colorContributions[iteration] = attributes.x;
            }
            else 
            {
               if( (*sceneInfo).pathTracingIteration>=NB_MAX_ITERATIONS && !BRDF )
               {
                   // Compute the BRDF for this ray (assuming Lambertian reflection)
                   BRDF=true;
                   Vertex O_E = rayOrigin.origin - closestIntersection;
                   /*
                   vectorReflection( rayOrigin.direction, O_E, normal );
                   reflectedTarget = closestIntersection - rayOrigin.direction;
                   */
                   int t=(3+index+(*sceneInfo).misc.y)%((*sceneInfo).size.x*(*sceneInfo).size.y);
                   reflectedTarget.x = normal.x+80000.f*randoms[t  ];
                   reflectedTarget.y = normal.y+80000.f*randoms[t+1];
                   reflectedTarget.z = normal.z+80000.f*randoms[t+2];
                   float cos_theta = dot(normalize(reflectedTarget),normal);
                   //float reflectance=0.1f;
                   //float BDRF = 2.f*reflectance*cos_theta;
                   colorContributions[iteration] = 0.5f*fabs(cos_theta);                  
               }
               else
               {
                   // No more intersections with primitives -> skybox
                   carryon = false;
                   colorContributions[iteration] = 1.f;                   
               }
            }         
         }

         // Contribute to final color
         rBlinn /= (iteration+1);
         recursiveBlinn.x=(rBlinn.x>recursiveBlinn.x) ? rBlinn.x:recursiveBlinn.x;
         recursiveBlinn.y=(rBlinn.y>recursiveBlinn.y) ? rBlinn.y:recursiveBlinn.y;
         recursiveBlinn.z=(rBlinn.z>recursiveBlinn.z) ? rBlinn.z:recursiveBlinn.z;

         rayOrigin.origin    = closestIntersection+reflectedTarget*REBOUND_EPSILON; 
         rayOrigin.direction = reflectedTarget;

         // Noise management
         if( (*sceneInfo).pathTracingIteration != 0 && materials[primitives[closestPrimitive].materialId].color.w != 0.f)
         {
            // Randomize view
            float ratio = materials[primitives[closestPrimitive].materialId].color.w;
            ratio *= (attributes.y==0.f) ? 1000.f : 1.f;
            int rindex = 3*(*sceneInfo).misc.y + (*sceneInfo).pathTracingIteration;
            rindex = rindex%((*sceneInfo).size.x*(*sceneInfo).size.y);
            rayOrigin.direction.x += randoms[rindex  ]*ratio;
            rayOrigin.direction.y += randoms[rindex+1]*ratio;
            rayOrigin.direction.z += randoms[rindex+2]*ratio;
         }
      }
      else
      {
         // Background
         if( (*sceneInfo).skybox.y!=MATERIAL_NONE)
         {
            colors[iteration] = skyboxMapping(sceneInfo,materials,textures,&rayOrigin);
         }
         else
         {
            if( (*sceneInfo).parameters.y==1 )
            {
               Vertex normal = {0.f,1.f,0.f,0.f};
               Vertex dir = normalize(rayOrigin.direction-rayOrigin.origin);
               float angle = 0.5f-dot( normal, dir);
               angle = (angle>1.f) ? 1.f: angle;
               colors[iteration] = (1.f-angle)*(*sceneInfo).backgroundColor;
            }
            else
            {
               colors[iteration] = (*sceneInfo).backgroundColor;
            }
         }
         colorContributions[iteration] = 1.f;
      }
      iteration++;
   }

   if( (*sceneInfo).graphicsLevel>=3 && reflectedRays != -1 ) // TODO: Draft mode should only test (*sceneInfo).pathTracingIteration==iteration
   {
      Vertex areas = {0.f,0.f,0.f,0.f};
      // TODO: Dodgy implementation		
      if( intersectionWithPrimitives(
         sceneInfo,
         boundingBoxes, nbActiveBoxes,
         primitives, nbActivePrimitives,
         materials, textures,
         &reflectedRay,
         reflectedRays,  
         &closestPrimitive, &closestIntersection, 
         &normal, &areas, &colorBox, currentMaterialId) )
      {
         Vertex attributes;
         attributes.x=materials[primitives[closestPrimitive].materialId].reflection;
         float4 color = primitiveShader( 
            index,
            sceneInfo, postProcessingInfo,
            boundingBoxes, nbActiveBoxes, 
            primitives, nbActivePrimitives, 
            lightInformation, lightInformationSize, nbActiveLamps, 
            materials, textures, randoms, 
            reflectedRay.origin, &normal, closestPrimitive, 
            &closestIntersection, areas, &closestColor,
            iteration, &refractionFromColor, &shadowIntensity, &rBlinn, &attributes );
         colors[reflectedRays] += color*reflectedRatio;

         (*primitiveXYId).w = shadowIntensity*255;
      }
   }

   for( int i=iteration-2; i>=0; --i)
   {
      colors[i] = colors[i]*(1.f-colorContributions[i]) + colors[i+1]*colorContributions[i];
   }
   intersectionColor = colors[0];
   intersectionColor += recursiveBlinn;

   (*intersection) = closestIntersection;

   float len = length(firstIntersection - (*ray).origin);
   (*depthOfField) = len;
   if( closestPrimitive != -1 )
   {
      CONST Primitive* primitive=&primitives[closestPrimitive];
      if( materials[(*primitive).materialId].attributes.z == 1 ) // Wireframe
      {
         len = (*sceneInfo).viewDistance;
      }
   }

   // --------------------------------------------------
   // Background color
   // --------------------------------------------------
   float D1 = (*sceneInfo).viewDistance*0.95f;
   if( (*sceneInfo).misc.z==1 && len>D1)
   {
      float D2 = (*sceneInfo).viewDistance*0.05f;
      float a = len - D1;
      float b = 1.f-(a/D2);
      intersectionColor = intersectionColor*b + (*sceneInfo).backgroundColor*(1.f-b);
   }

   // Primitive information
   (*primitiveXYId).y = iteration;

   // Depth of field
   intersectionColor -= colorBox;

   saturateVector( &intersectionColor );
   return intersectionColor;
}

/*
________________________________________________________________________________

Standard renderer
________________________________________________________________________________
*/
__kernel void k_standardRenderer(
   const int2                     occupancyParameters,
   int                            device_split,
   int                            stream_split,
   CONST BoundingBox*    boundingBoxes, 
   int                            nbActiveBoxes,
   CONST Primitive*      primitives, 
   int                            nbActivePrimitives,
   CONST LightInformation*     lightInformation, 
   int                            lightInformationSize, 
   int                            nbActiveLamps,
   CONST Material*       materials,
   CONST BitmapBuffer*   textures,
   CONST RandomBuffer*   randoms,
   Vertex                         origin,
   Vertex                         direction,
   Vertex                         angles,
   const SceneInfo                sceneInfo,
   const PostProcessingInfo       postProcessingInfo,
   CONST PostProcessingBuffer* postProcessingBuffer,
   CONST PrimitiveXYIdBuffer*  primitiveXYIds)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y*sceneInfo.size.x+x;

   float dof = postProcessingInfo.param1;
   float4 color = {0.f,0.f,0.f,0.f};

   // Beware out of bounds error!
   // And only process pixels that need extra rendering
   if(index>=sceneInfo.size.x*sceneInfo.size.y/occupancyParameters.x ||
      (sceneInfo.pathTracingIteration>primitiveXYIds[index].y &&   // Still need to process iterations
      primitiveXYIds[index].w==0 &&                                 // Shadows? if so, compute soft shadows by randomizing light positions
      sceneInfo.pathTracingIteration>0 && 
      sceneInfo.pathTracingIteration<=NB_MAX_ITERATIONS)) return;

   Ray ray;
   ray.origin = origin;
   ray.direction = direction;

   Vertex rotationCenter = {0.f,0.f,0.f,0.f};
   if( sceneInfo.renderingType==vt3DVision)
   {
      rotationCenter = origin;
   }

   bool antialiasingActivated = (sceneInfo.misc.w == 2);

   if( sceneInfo.pathTracingIteration == 0 )
   {
      postProcessingBuffer[index].x = 0.f;
      postProcessingBuffer[index].y = 0.f;
      postProcessingBuffer[index].z = 0.f;
      postProcessingBuffer[index].w = 0.f;
   }
   if( postProcessingInfo.type!=ppe_depthOfField && sceneInfo.pathTracingIteration>=NB_MAX_ITERATIONS )
   {
      // Randomize view for natural depth of field
      float a=postProcessingInfo.param1/100000.f;
      int rindex;
      rindex = 3*(index+sceneInfo.misc.y);
      rindex = rindex%(sceneInfo.size.x*sceneInfo.size.y-3);
      ray.origin.x += randoms[rindex  ]*postProcessingBuffer[index].w*a;
      ray.origin.y += randoms[rindex+1]*postProcessingBuffer[index].w*a;
      ray.origin.z += randoms[rindex+2]*postProcessingBuffer[index].w*a;
   }

   Vertex intersection;

   if( sceneInfo.misc.w == 1 ) // Isometric 3D
   {
      ray.direction.x = ray.origin.z*0.001f*(float)(x - (sceneInfo.size.x/2));
      ray.direction.y = -ray.origin.z*0.001f*(float)(device_split+stream_split+y - (sceneInfo.size.y/2));
      ray.origin.x = ray.direction.x;
      ray.origin.y = ray.direction.y;
   }
   else
   {
      float ratio=(float)sceneInfo.size.x/(float)sceneInfo.size.y;
      float2 step;
      step.x=ratio*6400.f/(float)sceneInfo.size.x;
      step.y=6400.f/(float)sceneInfo.size.y;
      ray.direction.x = ray.direction.x - step.x*(float)(x - (sceneInfo.size.x/2));
      ray.direction.y = ray.direction.y + step.y*(float)(device_split+stream_split+y - (sceneInfo.size.y/2));
   }

   vectorRotation( &ray.origin, rotationCenter, angles );
   vectorRotation( &ray.direction, rotationCenter, angles );

   // Antialisazing
   float2 AArotatedGrid[4] =
   {
      {  3.f,  5.f },
      {  5.f, -3.f },
      { -3.f, -5.f },
      { -5.f,  3.f }
   };

   if( sceneInfo.pathTracingIteration>primitiveXYIds[index].y && sceneInfo.pathTracingIteration>0 && sceneInfo.pathTracingIteration<=NB_MAX_ITERATIONS ) return;

   Ray r=ray;
   if( antialiasingActivated )
   {
      for( int I=0; I<4; ++I )
      {
         r.direction.x = ray.direction.x + 1.f*AArotatedGrid[I].x;
         r.direction.y = ray.direction.y + 1.f*AArotatedGrid[I].y;
         float4 c = launchRay(
            index,
            boundingBoxes, nbActiveBoxes,
            primitives, nbActivePrimitives,
            lightInformation, lightInformationSize, nbActiveLamps,
            materials, textures, 
            randoms,
            &r, 
            &sceneInfo, &postProcessingInfo,
            &intersection,
            &dof,
            &primitiveXYIds[index]);
         color += c;
      }
   }
   else
   {
      r.direction.x = ray.direction.x + 1.f*AArotatedGrid[sceneInfo.pathTracingIteration%4].x;
      r.direction.y = ray.direction.y + 1.f*AArotatedGrid[sceneInfo.pathTracingIteration%4].y;
   }
   color += launchRay(
      index,
      boundingBoxes, nbActiveBoxes,
      primitives, nbActivePrimitives,
      lightInformation, lightInformationSize, nbActiveLamps,
      materials, textures, 
      randoms,
      &r, 
      &sceneInfo, &postProcessingInfo,
      &intersection,
      &dof,
      &primitiveXYIds[index]);

   if( sceneInfo.parameters.z==1 )
   {
      // Randomize light intensity
      int rindex = index;
      rindex = rindex%(sceneInfo.size.x*sceneInfo.size.y);
      color += sceneInfo.backgroundColor*randoms[rindex]*5.f;
   }

   if( antialiasingActivated )
   {
      color /= 5.f;
   }

   if( sceneInfo.pathTracingIteration == 0 )
   {
      postProcessingBuffer[index].w = dof;
   }

   if( sceneInfo.pathTracingIteration<=NB_MAX_ITERATIONS )
   {
      postProcessingBuffer[index].x = color.x;
      postProcessingBuffer[index].y = color.y;
      postProcessingBuffer[index].z = color.z;
   }
   else
   {
      postProcessingBuffer[index].x += color.x;
      postProcessingBuffer[index].y += color.y;
      postProcessingBuffer[index].z += color.z;
   }
}

/*
________________________________________________________________________________

Anaglyph Renderer
________________________________________________________________________________
*/
__kernel void k_anaglyphRenderer(
   const int2   occupancyParameters,
   int          device_split,
   int          stream_split,
   CONST BoundingBox* boundingBoxes, int nbActiveBoxes,
   CONST Primitive* primitives, int nbActivePrimitives,
   CONST LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
   CONST Material*    materials,
   CONST BitmapBuffer* textures,
   CONST RandomBuffer* randoms,
   Vertex        origin,
   Vertex        direction,
   Vertex        angles,
   const SceneInfo     sceneInfo,
   const PostProcessingInfo postProcessingInfo,
   CONST PostProcessingBuffer* postProcessingBuffer,
   CONST PrimitiveXYIdBuffer*  primitiveXYIds)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y*sceneInfo.size.x+x;

   // Beware out of bounds error!
   if( index>=sceneInfo.size.x*sceneInfo.size.y/occupancyParameters.x ) return;

   float focus = primitiveXYIds[sceneInfo.size.x*sceneInfo.size.y/2].x - origin.z;
   float eyeSeparation = sceneInfo.width3DVision*(focus/direction.z);

   Vertex rotationCenter = {0.f,0.f,0.f,0.f};
   if( sceneInfo.renderingType==vt3DVision)
   {
      rotationCenter = origin;
   }

   if( sceneInfo.pathTracingIteration == 0 )
   {
      postProcessingBuffer[index].x = 0.f;
      postProcessingBuffer[index].y = 0.f;
      postProcessingBuffer[index].z = 0.f;
      postProcessingBuffer[index].w = 0.f;
   }

   float dof = postProcessingInfo.param1;
   Vertex intersection;
   Ray eyeRay;

   float ratio=(float)sceneInfo.size.x/(float)sceneInfo.size.y;
   float2 step;
   step.x=4.f*ratio*6400.f/(float)sceneInfo.size.x;
   step.y=4.f*6400.f/(float)sceneInfo.size.y;

   // Left eye
   eyeRay.origin.x = origin.x + eyeSeparation;
   eyeRay.origin.y = origin.y;
   eyeRay.origin.z = origin.z;

   eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.size.x/2));
   eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.size.y/2));
   eyeRay.direction.z = direction.z;

   //vectorRotation( eyeRay.origin, rotationCenter, angles );
   vectorRotation( &eyeRay.direction, rotationCenter, angles );

   float4 colorLeft = launchRay(
      index,
      boundingBoxes, nbActiveBoxes,
      primitives, nbActivePrimitives,
      lightInformation, lightInformationSize, nbActiveLamps,
      materials, textures, 
      randoms,
      &eyeRay, 
      &sceneInfo, &postProcessingInfo,
      &intersection,
      &dof,
      &primitiveXYIds[index]);

   // Right eye
   eyeRay.origin.x = origin.x - eyeSeparation;
   eyeRay.origin.y = origin.y;
   eyeRay.origin.z = origin.z;

   eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.size.x/2));
   eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.size.y/2));
   eyeRay.direction.z = direction.z;

   //vectorRotation( eyeRay.origin, rotationCenter, angles );
   vectorRotation( &eyeRay.direction, rotationCenter, angles );

   float4 colorRight = launchRay(
      index,
      boundingBoxes, nbActiveBoxes,
      primitives, nbActivePrimitives,
      lightInformation, lightInformationSize, nbActiveLamps,
      materials, textures, 
      randoms,
      &eyeRay, 
      &sceneInfo, &postProcessingInfo,
      &intersection,
      &dof,
      &primitiveXYIds[index]);

   float r1 = colorLeft.x*0.299f + colorLeft.y*0.587f + colorLeft.z*0.114f;
   float b1 = 0.f;
   float g1 = 0.f;

   float r2 = 0.f;
   float g2 = colorRight.y;
   float b2 = colorRight.z;

   if( sceneInfo.pathTracingIteration == 0 ) postProcessingBuffer[index].w = dof;
   if( sceneInfo.pathTracingIteration<=NB_MAX_ITERATIONS )
   {
      postProcessingBuffer[index].x = r1+r2;
      postProcessingBuffer[index].y = g1+g2;
      postProcessingBuffer[index].z = b1+b2;
   }
   else
   {
      postProcessingBuffer[index].x += r1+r2;
      postProcessingBuffer[index].y += g1+g2;
      postProcessingBuffer[index].z += b1+b2;
   }
}

/*
________________________________________________________________________________

3D Vision Renderer
________________________________________________________________________________
*/
__kernel void k_3DVisionRenderer(
   const int2   occupancyParameters,
   int          device_split,
   int          stream_split,
   CONST BoundingBox* boundingBoxes, int nbActiveBoxes,
   CONST Primitive* primitives, int nbActivePrimitives,
   CONST LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
   CONST Material*    materials,
   CONST BitmapBuffer* textures,
   CONST RandomBuffer* randoms,
   Vertex        origin,
   Vertex        direction,
   Vertex        angles,
   const SceneInfo     sceneInfo,
   const PostProcessingInfo postProcessingInfo,
   CONST PostProcessingBuffer* postProcessingBuffer,
   CONST PrimitiveXYIdBuffer*  primitiveXYIds)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y*sceneInfo.size.x+x;

   // Beware out of bounds error!
   if( index>=sceneInfo.size.x*sceneInfo.size.y/occupancyParameters.x ) return;

   float focus = primitiveXYIds[sceneInfo.size.x*sceneInfo.size.y/2].x - origin.z;
   float eyeSeparation = sceneInfo.width3DVision*(direction.z/focus);

   Vertex rotationCenter = {0.f,0.f,0.f,0.f};
   if( sceneInfo.renderingType==vt3DVision)
   {
      rotationCenter = origin;
   }

   if( sceneInfo.pathTracingIteration == 0 )
   {
      postProcessingBuffer[index].x = 0.f;
      postProcessingBuffer[index].y = 0.f;
      postProcessingBuffer[index].z = 0.f;
      postProcessingBuffer[index].w = 0.f;
   }

   float dof = postProcessingInfo.param1;
   Vertex intersection;
   int halfWidth  = sceneInfo.size.x/2;

   float ratio=(float)sceneInfo.size.x/(float)sceneInfo.size.y;
   float2 step;
   step.x=ratio*6400.f/(float)sceneInfo.size.x;
   step.y=6400.f/(float)sceneInfo.size.y;

   Ray eyeRay;
   if( x<halfWidth ) 
   {
      // Left eye
      eyeRay.origin.x = origin.x + eyeSeparation;
      eyeRay.origin.y = origin.y;
      eyeRay.origin.z = origin.z;

      eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.size.x/2) + halfWidth/2 ) + sceneInfo.width3DVision;
      eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.size.y/2));
      eyeRay.direction.z = direction.z;
   }
   else
   {
      // Right eye
      eyeRay.origin.x = origin.x - eyeSeparation;
      eyeRay.origin.y = origin.y;
      eyeRay.origin.z = origin.z;

      eyeRay.direction.x = direction.x - step.x*(float)(x - (sceneInfo.size.x/2) - halfWidth/2) - sceneInfo.width3DVision;
      eyeRay.direction.y = direction.y + step.y*(float)(y - (sceneInfo.size.y/2));
      eyeRay.direction.z = direction.z;
   }

   if(sqrt(eyeRay.direction.x*eyeRay.direction.x+eyeRay.direction.y*eyeRay.direction.y)>(halfWidth*6)) return;

   vectorRotation( &eyeRay.origin,    rotationCenter, angles );
   vectorRotation( &eyeRay.direction, rotationCenter, angles );

   float4 color = launchRay(
      index,
      boundingBoxes, nbActiveBoxes,
      primitives, nbActivePrimitives,
      lightInformation, lightInformationSize, nbActiveLamps,
      materials, textures, 
      randoms,
      &eyeRay, 
      &sceneInfo, &postProcessingInfo,
      &intersection,
      &dof,
      &primitiveXYIds[index]);

   // Randomize light intensity
   int rindex = index;
   rindex = rindex%(sceneInfo.size.x*sceneInfo.size.y);
   color += sceneInfo.backgroundColor*randoms[rindex]*5.f;

   // Contribute to final image
   if( sceneInfo.pathTracingIteration == 0 ) postProcessingBuffer[index].w = dof;
   if( sceneInfo.pathTracingIteration<=NB_MAX_ITERATIONS )
   {
      postProcessingBuffer[index].x = color.x;
      postProcessingBuffer[index].y = color.y;
      postProcessingBuffer[index].z = color.z;
   }
   else
   {
      postProcessingBuffer[index].x += color.x;
      postProcessingBuffer[index].y += color.y;
      postProcessingBuffer[index].z += color.z;
   }
}

/*
________________________________________________________________________________

3D Vision Renderer
________________________________________________________________________________
*/
__kernel void k_fishEyeRenderer(
   const int2   occupancyParameters,
   int          device_split,
   int          stream_split,
   CONST BoundingBox* boundingBoxes, int nbActiveBoxes,
   CONST Primitive* primitives, int nbActivePrimitives,
   CONST LightInformation* lightInformation, int lightInformationSize, int nbActiveLamps,
   CONST Material*    materials,
   CONST BitmapBuffer* textures,
   CONST RandomBuffer* randoms,
   Vertex        origin,
   Vertex        direction,
   Vertex        angles,
   const SceneInfo     sceneInfo,
   const PostProcessingInfo postProcessingInfo,
   CONST PostProcessingBuffer* postProcessingBuffer,
   CONST PrimitiveXYIdBuffer*  primitiveXYIds)
{
}

/*
________________________________________________________________________________

Post Processing Effect: Default
________________________________________________________________________________
*/
__kernel void k_default(
   const int2                     occupancyParameters,
   SceneInfo                      sceneInfo,
   PostProcessingInfo             PostProcessingInfo,
   CONST PostProcessingBuffer* postProcessingBuffer,
   __global BitmapBuffer*         bitmap) 
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y*sceneInfo.size.x+x;

   // Beware out of bounds error!
   if( index>=sceneInfo.size.x*sceneInfo.size.y/occupancyParameters.x ) return;

   float4 localColor = postProcessingBuffer[index];
   if(sceneInfo.pathTracingIteration>NB_MAX_ITERATIONS)
   {
      localColor /= (float)(sceneInfo.pathTracingIteration-NB_MAX_ITERATIONS+1);
   }
   makeColor( &sceneInfo, &localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Depth of field
________________________________________________________________________________
*/
__kernel void k_depthOfField(
   const int2                     occupancyParameters,
   SceneInfo                      sceneInfo,
   PostProcessingInfo             postProcessingInfo,
   CONST PostProcessingBuffer* postProcessingBuffer,
   CONST RandomBuffer*         randoms,
   __global BitmapBuffer*         bitmap) 
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y*sceneInfo.size.x+x;

   // Beware out of bounds error!
   if( index>=sceneInfo.size.x*sceneInfo.size.y/occupancyParameters.x ) return;

   float4 localColor = {0.f,0.f,0.f,0.f};
   float  depth=fabs(postProcessingBuffer[index].w-postProcessingInfo.param1)/sceneInfo.viewDistance;
   int    wh = sceneInfo.size.x*sceneInfo.size.y;

   for( int i=0; i<postProcessingInfo.param3; ++i )
   {
      int ix = i%wh;
      int iy = (i+100)%wh;
      int xx = x+depth*randoms[ix]*postProcessingInfo.param2;
      int yy = y+depth*randoms[iy]*postProcessingInfo.param2;
      if( xx>=0 && xx<sceneInfo.size.x && yy>=0 && yy<sceneInfo.size.y )
      {
         int localIndex = yy*sceneInfo.size.x+xx;
         if( localIndex>=0 && localIndex<wh )
         {
            localColor += postProcessingBuffer[localIndex];
         }
      }
      else
      {
         localColor += postProcessingBuffer[index];
      }
   }
   localColor /= postProcessingInfo.param3;

   if(sceneInfo.pathTracingIteration>NB_MAX_ITERATIONS)
      localColor /= (float)(sceneInfo.pathTracingIteration-NB_MAX_ITERATIONS+1);

   localColor.w = 1.f;

   makeColor( &sceneInfo, &localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Ambiant Occlusion
________________________________________________________________________________
*/
__kernel void k_ambientOcclusion(
   const int2                     occupancyParameters,
   SceneInfo                      sceneInfo,
   PostProcessingInfo             postProcessingInfo,
   CONST PostProcessingBuffer* postProcessingBuffer,
   CONST RandomBuffer*         randoms,
   __global BitmapBuffer*         bitmap) 
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y*sceneInfo.size.x+x;
   // Beware out of bounds error!
   if( index>=sceneInfo.size.x*sceneInfo.size.y/occupancyParameters.x ) return;

   int    wh = sceneInfo.size.x*sceneInfo.size.y;
   float occ = 0.f;
   float4 localColor = postProcessingBuffer[index];
   float  depth = localColor.w;
   const int step = 16;
   float c=0.f;
   int i=0;
   for( int X=-step; X<step; X+=2 )
   {
      for( int Y=-step; Y<step; Y+=2 )
      {
         int ix = i%wh;
         int iy = (i+100)%wh;
         ++i;
         c+=1.f;
         int xx = x+(X*postProcessingInfo.param2*randoms[ix]/10.f);
         int yy = y+(Y*postProcessingInfo.param2*randoms[iy]/10.f);
         if( xx>=0 && xx<sceneInfo.size.x && yy>=0 && yy<sceneInfo.size.y )
         {
            int localIndex = yy*sceneInfo.size.x+xx;
            if( postProcessingBuffer[localIndex].w<depth)
            {
               occ += 1.f-(postProcessingBuffer[localIndex].w-depth)/sceneInfo.viewDistance;
            }
         }
         else
         {
            occ += 1.f;
         }
      }
   }
   occ /= 5.f*c;
   //occ += 0.3f; // Ambient light
   occ = (occ>1.f) ? 1.f : occ;
   occ = (occ<0.f) ? 0.f : occ;
   if(sceneInfo.pathTracingIteration>NB_MAX_ITERATIONS)
   {
      localColor /= (float)(sceneInfo.pathTracingIteration-NB_MAX_ITERATIONS+1);
   }
   localColor.x -= occ;
   localColor.y -= occ;
   localColor.z -= occ;
   saturateVector( &localColor );

   localColor.w = 1.f;

   makeColor( &sceneInfo, &localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: radiosity
________________________________________________________________________________
*/
__kernel void k_radiosity(
   const int2                     occupancyParameters,
   SceneInfo                      sceneInfo,
   PostProcessingInfo             postProcessingInfo,
   CONST PrimitiveXYIdBuffer*  primitiveXYIds,
   CONST PostProcessingBuffer* postProcessingBuffer,
   CONST RandomBuffer*         randoms,
   __global BitmapBuffer*         bitmap) 
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y*sceneInfo.size.x+x;

   // Beware out of bounds error!
   if( index>=sceneInfo.size.x*sceneInfo.size.y/occupancyParameters.x ) return;

   int wh = sceneInfo.size.x*sceneInfo.size.y;

   float div = (sceneInfo.pathTracingIteration>NB_MAX_ITERATIONS) ? (float)(sceneInfo.pathTracingIteration-NB_MAX_ITERATIONS+1) : 1.f;

   float4 localColor = {0.f,0.f,0.f,0.f};
   for( int i=0; i<postProcessingInfo.param3; ++i )
   {
      int ix = (i+sceneInfo.misc.y+sceneInfo.pathTracingIteration)%wh;
      int iy = (i+sceneInfo.misc.y+sceneInfo.size.x)%wh;
      int xx = x+randoms[ix]*postProcessingInfo.param2;
      int yy = y+randoms[iy]*postProcessingInfo.param2;
      localColor += postProcessingBuffer[index];
      if( xx>=0 && xx<sceneInfo.size.x && yy>=0 && yy<sceneInfo.size.y )
      {
         int localIndex = yy*sceneInfo.size.x+xx;
         localColor += div*primitiveXYIds[localIndex].z/255.f;
      }
   }
   localColor /= postProcessingInfo.param3;
   localColor /= div;
   localColor.w = 1.f;
   makeColor( &sceneInfo, &localColor, bitmap, index ); 
}

/*
________________________________________________________________________________

Post Processing Effect: Default
________________________________________________________________________________
*/
__kernel void k_contrast(
   const int2                     occupancyParameters,
   SceneInfo                      sceneInfo,
   PostProcessingInfo             PostProcessingInfo,
   CONST PostProcessingBuffer* postProcessingBuffer,
   __global BitmapBuffer*         bitmap) 
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   int index = y*sceneInfo.size.x+x;
   // Beware out of bounds error!
   if( index>=sceneInfo.size.x*sceneInfo.size.y/occupancyParameters.x ) return;

   float4 localColor = postProcessingBuffer[index];
   const int step = 8;
   int c=0;
   float4 color={0.f,0.f,0.f,0.f};
   for( int X=-step; X<step; ++X )
   {
      for( int Y=-step; Y<step; ++Y )
      {
         if( X!=0 || Y!=0 )
         {
            int xx = x+X;
            int yy = y+Y;
            if( xx>=0 && xx<sceneInfo.size.x && yy>=0 && yy<sceneInfo.size.y )
            {
               int localIndex = yy*sceneInfo.size.x+xx;
               color += max(localColor,postProcessingBuffer[localIndex]);
               ++c;
            }
         }
      }
   }
   color/=c;
   localColor=(localColor*0.5f+color*0.5f);

   if(sceneInfo.pathTracingIteration>NB_MAX_ITERATIONS)
   {
      localColor /= (float)(sceneInfo.pathTracingIteration-NB_MAX_ITERATIONS+1);
   }
   saturateVector( &localColor );

   localColor.w = 1.f;

   makeColor( &sceneInfo, &localColor, bitmap, index ); 
}

