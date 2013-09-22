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

#include <GL/freeglut.h>
#include <iostream>
#include <time.h>

#include "../Logging.h"
#include "rtgl.h"

GenericGPUKernel* RayTracer::gKernel = nullptr;

SceneInfo gSceneInfo;
PostProcessingInfo gPostProcessingInfo;
const int DEFAULT_LIGHT_MATERIAL      = 1029;
const int gTotalPathTracingIterations = 1;
int4      gMisc = {otOpenGL,0,0,0};
float3    gRotationCenter = { 0.f, 0.f, 0.f };
float     gScale=1.f;

static bool ARB_multitexture_supported = false;
static bool EXT_texture_env_combine_supported = false;
static bool NV_register_combiners_supported = false;
static bool SGIX_depth_texture_supported = false;
static bool SGIX_shadow_supported = false;
static bool EXT_blend_minmax_supported = false;

void RayTracer::InitializeRaytracer( const int width, const int height, const bool initializeMaterials )
{
   // Scene
   gSceneInfo.width.x = width;
   gSceneInfo.height.x = height; 
   gSceneInfo.graphicsLevel.x = 4;
   gSceneInfo.nbRayIterations.x = 10;
   gSceneInfo.transparentColor.x =  0.f;
   gSceneInfo.viewDistance.x = 1000000.f;
   gSceneInfo.shadowIntensity.x = 0.8f;
   gSceneInfo.width3DVision.x = 400.f;
   gSceneInfo.backgroundColor.x = 1.f;
   gSceneInfo.backgroundColor.y = 1.f;
   gSceneInfo.backgroundColor.z = 1.f;
   gSceneInfo.backgroundColor.w = 0.f;
   gSceneInfo.renderingType.x = vtStandard;
   gSceneInfo.renderBoxes.x = 0;
   gSceneInfo.pathTracingIteration.x = 0;
   gSceneInfo.maxPathTracingIterations.x = gTotalPathTracingIterations;
   gSceneInfo.misc = gMisc;

   gPostProcessingInfo.type.x   = ppe_none;
   gPostProcessingInfo.param1.x = 10000.f;
   gPostProcessingInfo.param2.x = 1000.f;
   gPostProcessingInfo.param3.x = 200;

   // Kernel
   gKernel = new GenericGPUKernel(0, 480, 1, 0);
   gSceneInfo.pathTracingIteration.x = 0; 
   gKernel->setSceneInfo( gSceneInfo );
   gKernel->initBuffers();

   if( initializeMaterials )
   {
      createRandomMaterials(false,false);
   }
}
/*
________________________________________________________________________________

Create Random Materials
________________________________________________________________________________
*/
void RayTracer::createRandomMaterials( bool update, bool lightsOnly )
{
	srand(static_cast<int>(time(nullptr)));
   int start(0);
   int end(NB_MAX_MATERIALS);
   if( lightsOnly )
   {
      start = 120;
      end   = 130;
   }
	// Materials
	for( int i(start); i<end; ++i ) 
	{
		float4 specular = {0.f,0.f,0.f,0.f};
		specular.x = 0.5f;
		specular.y = 50.f;
		specular.z = 0.f;
		specular.w = 0.f;

		float reflection   = 0.f;
		float refraction   = 0.f;
		float transparency = 0.f;
		int   textureId = TEXTURE_NONE;
      float3 innerIllumination = { 0.f, 40000.f, gSceneInfo.viewDistance.x };
		bool procedural = false;
		bool wireframe = false;
		int  wireframeDepth = 0;
		float r,g,b,noise;
       
		r = rand()%1000/1000.f;
		g = rand()%1000/1000.f;
		b = rand()%1000/1000.f;
      noise = 0.f;

		switch( i )
		{
      case   0: 
      case   1: 
         {
            switch(rand()%3)
            {
            case 0: reflection=1.0f; refraction=1.66f; transparency=0.7f; break;
            case 1: reflection=0.8f; break;
            }
         }
         break;

      // Sky Box  
		case 101: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 0; break; 
		case 102: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 1; break; 
		case 103: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 2; break; 
		case 104: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 3; break; 
		case 105: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 4; break; 
		case 106: r=1.f; g=1.f; b=1.f; wireframe=true; textureId = 5; break; 
      
      // Cornell Box
      case 107: r=127.f/255.f; g=127.f/255.f; b=127.f/255.f; specular.x = 0.2f; specular.y = 10.f;  specular.w = 0.3f; break;
      case 108: r=154.f/255.f; g=94.f/255.f;  b=64.f/255.f;  specular.x = 0.1f; specular.y = 100.f; specular.w = 0.1f; break;
		case 109: r=92.f/255.f;  g=93.f/255.f;  b=150.f/255.f; specular.x = 0.3f; specular.y = 20.f;  specular.w = 0.5f; break;
		case 110: r=92.f/255.f;  g=150.f/255.f; b=93.f/255.f;  specular.x = 0.3f; specular.y = 20.f;  specular.w = 0.5f; break;
		
      // Fractals
      case 111: r=127.f/255.f; g=127.f/255.f; b=127.f/255.f; specular.x = 0.2f; specular.y = 10.f;  specular.w = 0.3f; wireframe=false; textureId=TEXTURE_MANDELBROT; break;
      case 112: r=154.f/255.f; g=94.f/255.f;  b=64.f/255.f;  specular.x = 0.1f; specular.y = 100.f; specular.w = 0.1f; wireframe=false; textureId=TEXTURE_JULIA; break;

      // Basic reflection
      case 113: /*r=0.5f; g=1.0f; b=0.7f; */reflection = 0.5f; refraction=1.6f; transparency=0.7f; break;
		case 114: /*r=1.f; g=1.f; b=1.f;*/ reflection = 0.9f; break;
      case 115: r=0.5f; g=1.0f; b=0.7f; reflection = 0.f; textureId = 0; break;
      case 116: /*r=0.f; g=0.f; b=0.f;*/ reflection = 0.1f; refraction=1.66f; transparency=0.5f; specular.x = 0.5f; specular.y = 10.f;break;
		case 117: r=1.f; g=0.f; b=0.f; reflection = 0.5f; break;
		case 118: r=0.f; g=1.f; b=1.f; reflection = 0.5f; break;

      // White
      case 119: r=1.f; g=1.f; b=1.f; break;

      // Wireframe
      case 120: innerIllumination.x=.5f; break; 
      case 121: innerIllumination.x=.5f; break; 
      case 122: innerIllumination.x=.5f; break; 
      case 123: innerIllumination.x=.5f; break; 
		case 124: innerIllumination.x=.5f; break; 
		case 125: innerIllumination.x=.5f; break; 
		case 126: innerIllumination.x=.5f; break; 
		case 127: innerIllumination.x=.5f; break; 
		case 128: innerIllumination.x=.5f; break; 
		case DEFAULT_LIGHT_MATERIAL: r=1.f; g=1.f; b=1.f; innerIllumination.x=1.f; break; 
      }

      int material = update ? i : RayTracer::gKernel->addMaterial();
		RayTracer::gKernel->setMaterial(
			material, r, g, b, noise,
			reflection, refraction, procedural, 
			wireframe, wireframeDepth,
			transparency, textureId,
			specular.x, specular.y, specular.w, 
         innerIllumination.x, innerIllumination.y, innerIllumination.z,
			false);
	}
   RayTracer::gKernel->compactBoxes(false);
}

void RayTracer::glBegin( GLint mode )
{
   //::glBegin(mode);
   RayTracer::gKernel->setGLMode(mode);
}

int RayTracer::glEnd()
{
   //::glEnd();
   return RayTracer::gKernel->setGLMode(-1);
}

void RayTracer::glEnable (GLenum cap)
{
   switch( cap )
   {
   case GL_LIGHTING: 
      {
         int p = RayTracer::gKernel->addPrimitive(ptSphere);
         RayTracer::gKernel->setPrimitive(p,-4.f*gScale,20.f*gScale,-10.f*gScale,1.f*gScale,0.f,0.f,DEFAULT_LIGHT_MATERIAL);
         //RayTracer::gKernel->compactBoxes(true);
         LOG_INFO(1, "[OpenGL] Light Added" );
      }
      break;
   }
   //::glEnable(cap);
}

void RayTracer::glDisable (GLenum cap)
{
   ::glDisable(cap);
}

void RayTracer::glClear (GLbitfield mask)
{
   ::glClear(mask);
}

void RayTracer::glFlush (void)
{
   ::glFlush();
}

void RayTracer::glVertex3f( GLfloat x, GLfloat y, GLfloat z )
{
   //::glVertex3f(x,y,z);
   RayTracer::gKernel->addVertex(x*gScale,y*gScale,z*gScale);
}

void RayTracer::glVertex3fv( const GLfloat* v )
{
   //::glVertex3f(x,y,z);
   RayTracer::gKernel->addVertex(v[0]*gScale,v[1]*gScale,v[2]*gScale);
}

void RayTracer::glNormal3f( GLfloat x, GLfloat y, GLfloat z )
{
   //::glNormal3f(x,y,z);
   RayTracer::gKernel->addNormal(x,y,z);
}

void RayTracer::glNormal3fv( const GLfloat* n )
{
   RayTracer::gKernel->addNormal(n[0],n[1],n[2]);
}

void RayTracer::glColor3f (GLfloat red, GLfloat green, GLfloat blue)
{
   ::glColor3f(red,green,blue);
}

void RayTracer::glRasterPos2f (GLfloat x, GLfloat y)
{
   ::glRasterPos2f(x,y);
}

void RayTracer::glTexParameterf (GLenum target, GLenum pname, GLfloat param)
{
   ::glTexParameterf(target,pname,param);
}

void RayTracer::glTexCoord2f (GLfloat s, GLfloat t)
{
   //::glTexCoord2f(s,t);
   RayTracer::gKernel->addTextCoord(s,t,0.f);
}

void RayTracer::glTexCoord3f( GLfloat x, GLfloat y, GLfloat z )
{
   //::glTexCoord3f(x,y,z);
   RayTracer::gKernel->addTextCoord(x,y,z);
}

void RayTracer::glTexEnvf (GLenum target, GLenum pname, GLfloat param)
{
   ::glTexEnvf(target,pname,param);
}

void RayTracer::glTexImage2D (GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *pixels)
{
   ::glTexImage2D(target,level,internalformat,width,height,border,format,type,pixels);
}

/*
* GLUT
*/

void RayTracer::glutInit( int* pargc, char** argv )
{
   ::glutInit(pargc, argv);
}

void RayTracer::glutInitWindowPosition( int x, int y )
{
   ::glutInitWindowPosition(x,y);
}

void RayTracer::glViewport(int a, int b, int width, int height )
{
   // OpenGL
   ARB_multitexture_supported=false;
   EXT_texture_env_combine_supported=false;
   NV_register_combiners_supported=false;
   SGIX_depth_texture_supported=false;
   SGIX_shadow_supported=false;
   EXT_blend_minmax_supported=false;

   // Initialize Raytracer
   InitializeRaytracer(width, height, true);
   //::glViewport(a,b,width,height);
}

void RayTracer::glutInitWindowSize( int width, int height )
{
   // Initialize Raytracer
   InitializeRaytracer(width, height, false);

   ::glutInitWindowSize(width,height);
}

void RayTracer::glutInitDisplayMode( unsigned int displayMode )
{
   ::glutInitDisplayMode(displayMode);
}

void RayTracer::glutMainLoop( void )
{
   ::glutMainLoop();
}

int  RayTracer::glutCreateWindow( const char* title )
{
   return ::glutCreateWindow(title);
}

void RayTracer::glutDestroyWindow( int window )
{
   ::glutDestroyWindow(window);
   delete gKernel;
   gKernel = nullptr;
}

void RayTracer::glutFullScreen( void )
{
   ::glutFullScreen();
}

void RayTracer::glLoadIdentity()
{
   float3 eye = {0.f,0.f, -30.f*gScale};
   float3 dir = {0.f,0.f, -30.f*gScale+10000.f};
   float3 angles = {0.f,0.f, 0.f};
   RayTracer::gKernel->setCamera(eye,dir,angles);
   RayTracer::gKernel->setSceneInfo(gSceneInfo);
   RayTracer::gKernel->setPostProcessingInfo(gPostProcessingInfo);
   RayTracer::gKernel->compactBoxes(true);
   RayTracer::gKernel->render_begin(0);
   RayTracer::gKernel->render_end();
   RayTracer::gKernel->resetFrame();
}

int  RayTracer::glutGet( GLenum query )
{
   return ::glutGet(query);
}

int  RayTracer::glutDeviceGet( GLenum query )
{
   return ::glutDeviceGet(query);
}

int  RayTracer::glutGetModifiers( void )
{
   return ::glutGetModifiers();
}

int  RayTracer::glutLayerGet( GLenum query )
{
   return ::glutLayerGet(query);
}

void RayTracer::glutKeyboardFunc( void (* callback)( unsigned char, int, int ) )
{
   ::glutKeyboardFunc(callback);
}

void RayTracer::glutDisplayFunc( void (* callback)( void ) )
{
   ::glutDisplayFunc(callback);
}

void RayTracer::glutMouseFunc( void (* callback)( int, int, int, int ) )
{
   ::glutMouseFunc(callback);
}

void RayTracer::glutMotionFunc( void (* callback)( int, int ) )
{
   ::glutMotionFunc(callback);
}

void RayTracer::glutTimerFunc( unsigned int time, void (* callback)( int ), int value )
{
   ::glutTimerFunc(time,callback,value);
}

int  RayTracer::glutCreateMenu( void(* callback)( int menu ) )
{
   return ::glutCreateMenu(callback);
}

void RayTracer::glutDestroyMenu( int menu )
{
   ::glutDestroyMenu(menu);
}

void RayTracer::glutAddMenuEntry( const char* label, int value )
{
   ::glutAddMenuEntry(label,value);
}

void RayTracer::glutAttachMenu( int button )
{
   ::glutAttachMenu(button);
}

void RayTracer::glutBitmapString( void* font, const unsigned char *string )
{
   ::glutBitmapString(font,string);
}

void RayTracer::glutPostRedisplay( void )
{
   ::glutPostRedisplay();
}

void RayTracer::glutSwapBuffers( void )
{
   ::glutSwapBuffers();
}

void RayTracer::gluSphere(void *, GLfloat radius, GLint , GLint)
{
   int p = RayTracer::gKernel->addPrimitive(ptSphere);
   RayTracer::gKernel->setPrimitive(p, 0.f, 0.f, 0.f, radius*gScale, 0.f, 0.f, 1 );
}

GLUquadricObj* RayTracer::gluNewQuadric(void)
{
   return 0;
}

void RayTracer::glClearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
   gSceneInfo.backgroundColor.x = red;
   gSceneInfo.backgroundColor.y = green;
   gSceneInfo.backgroundColor.z = blue;
   gSceneInfo.backgroundColor.w = alpha;
}
