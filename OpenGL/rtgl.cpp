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
const int DEFAULT_LIGHT_MATERIAL = NB_MAX_MATERIALS-1;
const int gTotalPathTracingIterations = 1;
int4      gMisc = {otOpenGL,0,0,0};
float3    gRotationCenter = { 0.f, 0.f, 0.f };
float     gScale=1.f;
// Current Material
int       gCurrentMaterial(0); 
bool      gLighting(false);

static bool ARB_multitexture_supported = false;
static bool EXT_texture_env_combine_supported = false;
static bool NV_register_combiners_supported = false;
static bool SGIX_depth_texture_supported = false;
static bool SGIX_shadow_supported = false;
static bool EXT_blend_minmax_supported = false;

// Utils
int RGBToInt(float r, float g, float b)
{
   int R=int(r*255.f) << 16;
   int G=int(g*255.f) << 8;
   int B=int(b*255.f);
   return (R+G+B);
}

void IntToRGB(int v, float& r, float& g, float& b)
{
   int V = v;
   int R=(V & 0x00FF0000) >> 16;
   int G=(V & 0x0000FF00) >> 8;
   int B=(V & 0x000000FF);

   r = R/255.f;
   g = G/255.f;
   b = B/255.f;
}

void RayTracer::InitializeRaytracer( const int width, const int height, const bool initializeMaterials )
{
   gCurrentMaterial = 0;
   // Scene
   gSceneInfo.width.x = width;
   gSceneInfo.height.x = height; 
   gSceneInfo.graphicsLevel.x = 4;
   gSceneInfo.nbRayIterations.x = 5;
   gSceneInfo.transparentColor.x =  0.f;
   gSceneInfo.viewDistance.x = 1000000.f;
   gSceneInfo.shadowIntensity.x = 0.9f;
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
   if(gKernel) delete gKernel;
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
   long R = 0;
   long G = 0;
   long B = 0;
	for( int i(start); i<end; ++i ) 
	{
		float4 specular = {0.f,0.f,0.f,0.f};
		specular.x = 0.5f;
		specular.y = 500.f;
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
      bool fastTransparency = false;

      /*
      int step(64);
      if( R<=255 )
      {
         r = R/255.f;
         g = G/255.f;
         b = B/255.f;

         B += step;
         if( B>255 )
         {
            B = 0;
            G += step;
            if( G>255 )
            {
               G = 0;
               R += step;
            }
         }
      }
      else
      */
      {
         r = 0.5f+(rand()%255)/512.f;
         g = 0.5f+(rand()%255)/512.f;
         b = 0.5f+(rand()%255)/512.f;
      }
      noise = 0.f;
		switch( i )
		{
      case   0: 
      case   1: 
         {
            switch(rand()%3)
            {
            case 0: reflection=1.0f; refraction=1.66f; transparency=0.5f; break;
            case 1: reflection=0.8f; break;
            }
         }
         break;
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
			fastTransparency);
	}
   //RayTracer::gKernel->compactBoxes(false);
}

void RayTracer::glBegin( GLint mode )
{
   RayTracer::gKernel->setGLMode(mode,gCurrentMaterial);
}

int RayTracer::glEnd()
{
   return RayTracer::gKernel->setGLMode(-1,gCurrentMaterial);
}

void RayTracer::glEnable (GLenum cap)
{
   switch( cap )
   {
   case GL_LIGHTING: 
      {
         if( !gLighting )
         {
            int p = RayTracer::gKernel->addPrimitive(ptSphere);
            RayTracer::gKernel->setPrimitive(p,-10.f*gScale,10.f*gScale,-10.f*gScale,0.1f*gScale,0.1f*gScale,0.1f*gScale,DEFAULT_LIGHT_MATERIAL);
            gLighting = true;
            //RayTracer::gKernel->compactBoxes(true);
            LOG_INFO(1, "[OpenGL] Light Added" );
         }
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
   /*
   int i(0);
   bool found(false);
   while( !found && i<gKernel->getNbActiveMaterials() )
   {
      Material* m = gKernel->getMaterial(i);
      if( m && (fabs(m->color.x - red)<0.1f && fabs(m->color.y - green)<0.1f && fabs(m->color.z - blue)<0.1f ) )
      {
         found = true;
      }
      else
      {
         ++i;
      }
   }
   gCurrentMaterial = found ? i : 0;
   */
}

void RayTracer::glColor4f(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
   RayTracer::glColor3f( red, green, blue );
}

void RayTracer::glRasterPos2f (GLfloat x, GLfloat y)
{
   ::glRasterPos2f(x,y);
}

void RayTracer::glTexParameterf (GLenum target, GLenum pname, GLfloat param)
{
   ::glTexParameterf(target,pname,param);
}

void RayTracer::glTexCoord2f (GLfloat s, GLfloat t )
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
   float3 eye = {0.f,0.f, -15.f*gScale};
   float3 dir = {0.f,0.f, -15.f*gScale+5000.f};
   float3 angles = {0.f,0.f, 0.f};
   RayTracer::gKernel->setCamera(eye,dir,angles);
   RayTracer::gKernel->setSceneInfo(gSceneInfo);
   RayTracer::gKernel->setPostProcessingInfo(gPostProcessingInfo);
   RayTracer::gKernel->compactBoxes(true);
   for( int i(0); i<gTotalPathTracingIterations; ++i)
   {
      RayTracer::gKernel->render_begin(0);
      RayTracer::gKernel->render_end();
   }
   RayTracer::gKernel->resetFrame();
   gLighting=false;
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
