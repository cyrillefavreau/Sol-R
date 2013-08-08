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

#include "rtgl.h"

#include <GL/freeglut.h>

#ifdef USE_CUDA
#include "../Cuda/CudaKernel.h"
#else
#include "../CPUKernel.h"
#endif // USE_CUDA

GPUKernel* RayTracer::gKernel = nullptr;

SceneInfo gSceneInfo;
PostProcessingInfo gPostProcessingInfo;
int       gTotalPathTracingIterations = 100;
int4      gMisc = {otOpenGL,0,1,0};
float3    gRotationCenter = { 0.f, 0.f, 0.f };


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
   ::glEnable(cap);
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
   RayTracer::gKernel->addVertex(x,y,z);
}

void RayTracer::glNormal3f( GLfloat x, GLfloat y, GLfloat z )
{
   //::glNormal3f(x,y,z);
   RayTracer::gKernel->addNormal(x,y,z);
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

void RayTracer::glutInitWindowSize( int width, int height )
{
   // Scene
   gSceneInfo.width.x = width;
   gSceneInfo.height.x = height; 
   gSceneInfo.graphicsLevel.x = 4;
   gSceneInfo.nbRayIterations.x = 3;
   gSceneInfo.transparentColor.x =  3.0f;
   gSceneInfo.viewDistance.x = 1000000.f;
   gSceneInfo.shadowIntensity.x = 0.2f;
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
#ifdef USE_CUDA
   gKernel = new CudaKernel(0, 480, 0, 0);
#else
   gKernel = new CPUKernel(0, 480, 0, 0);
#endif // USE_CUDA
   gSceneInfo.pathTracingIteration.x = 0; 
   gKernel->setSceneInfo( gSceneInfo );
   gKernel->initBuffers();

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
