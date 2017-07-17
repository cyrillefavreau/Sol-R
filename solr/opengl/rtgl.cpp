/* Copyright (c) 2011-2014, Cyrille Favreau
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
 *
 * This file is part of Sol-R <https://github.com/cyrillefavreau/Sol-R>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "../Consts.h"
#include "../Logging.h"
#include "rtgl.h"

#ifdef USE_CUDA
#include "../cuda/CudaKernel.h"
typedef CudaKernel GenericGPUKernel;
#else
#ifdef USE_OPENCL
#include "../opencl/OpenCLKernel.h"
typedef OpenCLKernel GenericGPUKernel;
#else
#include "../cpu/CPUKernel.h"
typedef CPUKernel GenericGPUKernel;
#endif // USE_OPENCL
#endif // USE_CUDA

GPUKernel *SolR::gKernel = 0;

SceneInfo gSceneInfo;
PostProcessingInfo glPostProcessingInfo;
const int gTotalPathTracingIterations = 1;
INT4 gMisc = {otOpenGL, 0, 0, 0};
Vertex gRotationCenter = {0.f, 0.f, 0.f};
float gScale = 1.0f;
// Current Material
int gCurrentTexture(-1);
bool gLighting(false);
// Camera. THIS IS UGLY
Vertex gEye = {gScale / 10.f, 0.f, -20.f * gScale};
Vertex gDir = {gScale / 10.f, 0.f, -20.f * gScale + 5000.f};
Vertex gAngles = {0.f, 0.f, 0.f};

#ifdef USE_OPENCL
// OpenCL
void SolR::setOpenCLPlatform(const int platform)
{
    SolR::gOpenCLPlatform = platform;
}

void SolR::setOpenCLDevice(const int device)
{
    SolR::gOpenCLDevice = device;
}
#endif // USE_OPENCL

void SolR::Initialize(const int width, const int height)
{
#ifdef USE_OPENCL
    LOG_INFO(1, "Intializing Raytracing engine: " << gOpenCLPlatform << "," << gOpenCLDevice);
#else
    LOG_INFO(1, "Intializing Raytracing engine");
#endif // USE_OPENCL
    if (!gKernel)
    {
        // Scene
        gSceneInfo.size.x = width;
        gSceneInfo.size.y = height;
        gSceneInfo.graphicsLevel = 4;
        gSceneInfo.nbRayIterations = 5;
        gSceneInfo.transparentColor = 0.f;
        gSceneInfo.viewDistance = 100000.f;
        gSceneInfo.shadowIntensity = 0.9f;
        gSceneInfo.width3DVision = 1.3f;
        gSceneInfo.backgroundColor.x = 1.f;
        gSceneInfo.backgroundColor.y = 1.f;
        gSceneInfo.backgroundColor.z = 1.f;
        gSceneInfo.backgroundColor.w = 0.f;
        gSceneInfo.renderingType = vtStandard;
        gSceneInfo.renderBoxes = 0;
        gSceneInfo.pathTracingIteration = 0;
        gSceneInfo.maxPathTracingIterations = gTotalPathTracingIterations;
        gSceneInfo.misc = gMisc;

        glPostProcessingInfo.type = ppe_none;
        glPostProcessingInfo.param1 = 10000.f;
        glPostProcessingInfo.param2 = 1000.f;
        glPostProcessingInfo.param3 = 200;

        // Kernel
        gKernel = new GenericGPUKernel(0, 480, gOpenCLPlatform, gOpenCLDevice);
        gSceneInfo.pathTracingIteration = 0;
        gKernel->setSceneInfo(gSceneInfo);
        gKernel->initBuffers();
        createRandomMaterials(false, false);
    }
}

void SolR::glCompactBoxes()
{
    if (SolR::gKernel)
    {
        SolR::gKernel->compactBoxes(true);
    }
}

/*
________________________________________________________________________________

Create Random Materials
________________________________________________________________________________
*/
void SolR::createRandomMaterials(bool update, bool lightsOnly)
{
    srand(static_cast<int>(time(0)));
    int start(0);
    int end(NB_MAX_MATERIALS);
    if (lightsOnly)
    {
        start = 120;
        end = 130;
    }
    // Materials
    long R = 0;
    long G = 0;
    long B = 0;
    for (int i(start); i < end; ++i)
    {
        FLOAT4 specular = {0.f, 0.f, 0.f, 0.f};
        specular.x = 1.f;
        specular.y = 500.f;
        specular.z = 0.f;
        specular.w = 0.f;

        float reflection = 0.f;
        float refraction = 0.f;
        float transparency = 0.f;
        int textureId = TEXTURE_NONE;
        Vertex innerIllumination = {0.f, 40000.f, gSceneInfo.viewDistance};
        bool procedural = false;
        bool wireframe = false;
        int wireframeDepth = 0;
        float r, g, b, noise;
        float opacity = gSceneInfo.viewDistance;
        bool fastTransparency = false;

        r = 0.5f + (rand() % 255) / 512.f;
        g = 0.5f + (rand() % 255) / 512.f;
        b = 0.5f + (rand() % 255) / 512.f;
        noise = 0.f;

        // if( i>0 && i<100 )
        {
            reflection = float(rand() % 11) / 10.f;
            refraction = 1.f + float(rand() % 11) / 10.f;
            transparency = float(rand() % 11) / 10.f;
            // reflection=1.f; refraction=1.66f; transparency=0.8f;
        }

        switch (i)
        {
        case DEFAULT_LIGHT_MATERIAL:
            r = 1.f;
            g = 1.f;
            b = 1.f;
            innerIllumination.x = 1.f;
            break;
        }

        int material = update ? i : SolR::gKernel->addMaterial();
        SolR::gKernel->setMaterial(material, r, g, b, noise, reflection, refraction, procedural, wireframe,
                                        wireframeDepth, transparency, opacity, textureId, TEXTURE_NONE, TEXTURE_NONE,
                                        TEXTURE_NONE, TEXTURE_NONE, TEXTURE_NONE, TEXTURE_NONE, specular.x, specular.y,
                                        specular.w, innerIllumination.x, innerIllumination.y, innerIllumination.z,
                                        fastTransparency);
    }
}

void SolR::glBegin(GLint mode)
{
    SolR::gKernel->setGLMode(mode);
}

int SolR::glEnd()
{
    return SolR::gKernel->setGLMode(-1);
}

void SolR::glEnable(GLenum cap)
{
    switch (cap)
    {
    case GL_LIGHTING:
    {
        if (!gLighting)
        {
            int p = SolR::gKernel->addPrimitive(ptSphere);
            SolR::gKernel->setPrimitive(p, 20.f * gScale, 20.f * gScale, -20.f * gScale, 0.1f * gScale,
                                             0.1f * gScale, 0.1f * gScale, DEFAULT_LIGHT_MATERIAL);
            gLighting = true;
            LOG_INFO(3, "[OpenGL] Light Added");
        }
    }
    break;
    }
    //::glEnable(cap);
}

void SolR::glDisable(GLenum cap)
{
    ::glDisable(cap);
}

void SolR::glClear(GLbitfield mask)
{
    ::glClear(mask);
}

void SolR::glVertex3f(GLfloat x, GLfloat y, GLfloat z)
{
    //::glVertex3f(x,y,z);
    SolR::gKernel->addVertex(x * gScale, y * gScale, z * gScale);
}

void SolR::glVertex3fv(const GLfloat *v)
{
    //::glVertex3f(x,y,z);
    SolR::gKernel->addVertex(v[0] * gScale, v[1] * gScale, v[2] * gScale);
}

void SolR::glNormal3f(GLfloat x, GLfloat y, GLfloat z)
{
    //::glNormal3f(x,y,z);
    SolR::gKernel->addNormal(x, y, z);
}

void SolR::glNormal3fv(const GLfloat *n)
{
    SolR::gKernel->addNormal(n[0], n[1], n[2]);
}

void SolR::glColor3f(GLfloat red, GLfloat green, GLfloat blue)
{
    glColor4f(red, green, blue, 0.f);
}

void SolR::glColor4f(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    /*
  bool found(false);
  unsigned int i(0);
  while( i<SolR::gKernel->getNbActiveMaterials() && !found )
  {
     Material* mat=SolR::gKernel->getMaterial(i);
     if (mat->color.x==red && mat->color.x==green && mat->color.x==blue &&
  mat->transparency.x==alpha)
     {
        found=true;
     }
     else
     {
        ++i;
     }
  }
  if( found )
  {
     SolR::gKernel->setMaterialColor(i,red,green,blue);
  }
  else
  */
    {
        int m = SolR::gKernel->getCurrentMaterial();
        ++m;
        SolR::gKernel->setMaterial(m, red, green, blue, 0.f, 0.f, 1.2f, false, 0.f, 0, alpha,
                                        gSceneInfo.viewDistance, MATERIAL_NONE, MATERIAL_NONE, MATERIAL_NONE,
                                        MATERIAL_NONE, TEXTURE_NONE, TEXTURE_NONE, TEXTURE_NONE, 1.f, 200.f, 1000.f,
                                        0.f, 0.f, 0.f, false);
        SolR::gKernel->setCurrentMaterial(m);
    }
}

void SolR::glRasterPos2f(GLfloat x, GLfloat y)
{
    ::glRasterPos2f(x, y);
}

void SolR::glRasterPos3f(GLfloat x, GLfloat y, GLfloat z)
{
    ::glRasterPos3f(x, y, z);
}

void SolR::glTexParameterf(GLenum target, GLenum pname, GLfloat param)
{
    ::glTexParameterf(target, pname, param);
}

void SolR::glTexCoord2f(GLfloat s, GLfloat t)
{
    //::glTexCoord2f(s,t);
    SolR::gKernel->addTextCoord(s, t, 0.f);
}

void SolR::glTexCoord3f(GLfloat x, GLfloat y, GLfloat z)
{
    //::glTexCoord3f(x,y,z);
    SolR::gKernel->addTextCoord(x, y, z);
}

void SolR::glTexEnvf(GLenum target, GLenum pname, GLfloat param)
{
    ::glTexEnvf(target, pname, param);
}

void SolR::glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height,
                             GLint border, GLenum format, GLenum type, const GLvoid *pixels)
{
    ::glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
}

/*
* GLUT
*/

void SolR::glutInit(int *pargc, char **argv)
{
    ::glutInit(pargc, argv);
}

void SolR::glutInitWindowPosition(int x, int y)
{
    ::glutInitWindowPosition(x, y);
}

void SolR::glutReshapeWindow(int width, int height)
{
    ::glutReshapeWindow(width, height);
}

void SolR::glViewport(int a, int b, int width, int height)
{
    // Initialize SolR
    Initialize(width, height);
    ::glViewport(a, b, width, height);
}

void SolR::glutInitWindowSize(int width, int height)
{
    // Initialize SolR
    Initialize(width, height);

    ::glutInitWindowSize(width, height);
}

void SolR::glutInitDisplayMode(unsigned int displayMode)
{
    ::glutInitDisplayMode(displayMode);
}

void SolR::glutMainLoop(void)
{
    ::glutMainLoop();
}

int SolR::glutCreateWindow(const char *title)
{
    return ::glutCreateWindow(title);
}

void SolR::glutDestroyWindow(int window)
{
    ::glutDestroyWindow(window);
    delete gKernel;
    gKernel = 0;
}

void SolR::glutFullScreen(void)
{
    ::glutFullScreen();
}

void SolR::glLoadIdentity()
{
    SolR::render();
    ::glLoadIdentity();
}

int SolR::glutGet(GLenum query)
{
    return ::glutGet(query);
}

int SolR::glutDeviceGet(GLenum query)
{
    return ::glutDeviceGet(query);
}

int SolR::glutGetModifiers(void)
{
    return ::glutGetModifiers();
}

int SolR::glutLayerGet(GLenum query)
{
    return ::glutLayerGet(query);
}

void SolR::glutKeyboardFunc(void (*callback)(unsigned char, int, int))
{
    ::glutKeyboardFunc(callback);
}

void SolR::glutDisplayFunc(void (*callback)(void))
{
    ::glutDisplayFunc(callback);
}

void SolR::glutMouseFunc(void (*callback)(int, int, int, int))
{
    ::glutMouseFunc(callback);
}

void SolR::glutMotionFunc(void (*callback)(int, int))
{
    ::glutMotionFunc(callback);
}

void SolR::glutTimerFunc(unsigned int time, void (*callback)(int), int value)
{
    ::glutTimerFunc(time, callback, value);
}

int SolR::glutCreateMenu(void (*callback)(int menu))
{
    return ::glutCreateMenu(callback);
}

void SolR::glutDestroyMenu(int menu)
{
    ::glutDestroyMenu(menu);
}

void SolR::glutAddMenuEntry(const char *label, int value)
{
    ::glutAddMenuEntry(label, value);
}

void SolR::glutAttachMenu(int button)
{
    ::glutAttachMenu(button);
}

void SolR::glutBitmapString(void *font, const unsigned char *string)
{
#ifndef __APPLE__
    //::glutBitmapString(font, string);
#endif
}

void SolR::glutPostRedisplay(void)
{
    ::glutPostRedisplay();
}

void SolR::glutSwapBuffers(void)
{
    ::glutSwapBuffers();
}

void SolR::gluSphere(void *, GLfloat radius, GLint, GLint)
{
    int p = SolR::gKernel->addPrimitive(ptSphere);
    Vertex translation = SolR::gKernel->getTranslation();
    int m = SolR::gKernel->getCurrentMaterial();
    SolR::gKernel->setPrimitive(p, translation.x * gScale, translation.y * gScale, translation.z * gScale,
                                     radius * gScale, 0.f, 0.f, m);
}

void SolR::glutWireSphere(GLdouble radius, GLint, GLint)
{
    int p = SolR::gKernel->addPrimitive(ptSphere);
    Vertex translation = SolR::gKernel->getTranslation();
    int m = SolR::gKernel->getCurrentMaterial();
    SolR::gKernel->setPrimitive(p, translation.x * gScale, translation.y * gScale, translation.z * gScale,
                                     static_cast<float>(radius) * gScale, 0.f, 0.f, m);
}

GLUquadricObj *SolR::gluNewQuadric(void)
{
    return 0;
}

void SolR::glClearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    gSceneInfo.backgroundColor.x = red;
    gSceneInfo.backgroundColor.y = green;
    gSceneInfo.backgroundColor.z = blue;
    gSceneInfo.backgroundColor.w = alpha;
}

void SolR::glMaterialfv(GLenum face, GLenum pname, const GLfloat *params)
{
    switch (pname)
    {
    case GL_AMBIENT:
        SolR::glColor3f(params[0], params[1], params[2]);
        break;
    }
}

void SolR::glGenTextures(GLsizei n, GLuint *textures)
{
    ++gCurrentTexture;
    SolR::gKernel->setTexturesTransfered(false);
    *textures = gCurrentTexture;
}

void SolR::glBindTexture(GLenum target, GLuint texture)
{
    switch (target)
    {
    case GL_TEXTURE_2D:
        SolR::gKernel->setMaterialTextureId(gCurrentTexture);
        break;
    }
}

int SolR::gluBuild2DMipmaps(GLenum target, GLint components, GLint width, GLint height, GLenum format, GLenum type,
                                 const void *data)
{
    TextureInformation textureInfo;
    textureInfo.size.x = width;
    textureInfo.size.y = height;

    switch (format)
    {
    case GL_RGB:
        textureInfo.size.z = 3;
        break;
    case GL_RGBA:
        textureInfo.size.z = 4;
        break;
    }
    textureInfo.offset = 0;
    textureInfo.buffer = (unsigned char *)data;

    SolR::gKernel->setTexture(gCurrentTexture, textureInfo);
    return 0;
}

void SolR::setAngles(GLfloat x, GLfloat y, GLfloat z)
{
    gAngles.x = x;
    gAngles.y = y;
    gAngles.z = z;
}

void SolR::glFlush()
{
    ::glFlush();
}

void SolR::glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height,
                                GLenum format, GLenum type, const GLvoid *data);

void SolR::glPushAttrib(GLbitfield mask)
{
}

void SolR::glPopAttrib()
{
}

void SolR::glTexParameteri(GLenum target, GLenum pname, GLint param)
{
}

void SolR::glBlendFunc(GLenum sfactor, GLenum dfactor)
{
}

void SolR::glMatrixMode(GLenum mode)
{
}

void SolR::glPushMatrix()
{
}

void SolR::glPopMatrix()
{
}

GLenum SolR::glGetError()
{
    return ::glGetError();
}

void SolR::glVertex2i(GLint x, GLint y)
{
    glVertex3f(static_cast<GLfloat>(x), static_cast<GLfloat>(y), 0.f);
}

void SolR::glOrtho(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble nearVal, GLdouble farVal)
{
    // gMisc.w = 1;
}

void SolR::render()
{
    if (SolR::gKernel)
    {
        Vertex rotation = SolR::gKernel->getRotation();
        SolR::gKernel->setCamera(gEye, gDir, rotation);
        SolR::gKernel->setSceneInfo(gSceneInfo);
        SolR::gKernel->setPostProcessingInfo(glPostProcessingInfo);
        if (!gLighting)
        {
            // if no light is defined, I add one
            int p = SolR::gKernel->addPrimitive(ptSphere);
            SolR::gKernel->setPrimitive(p, 20.f * gScale, 20.f * gScale, 20.f * gScale, 0.1f * gScale,
                                             0.1f * gScale, 0.1f * gScale, DEFAULT_LIGHT_MATERIAL);

            // p = SolR::gKernel->addPrimitive(ptSphere);
            // SolR::gKernel->setPrimitive(p,0.f,0.f,0.f,0.1f*gScale,0.1f*gScale,0.1f*gScale,0);
            gLighting = true;
        }

        SolR::gKernel->compactBoxes(false);
        for (int i(0); i < gTotalPathTracingIterations; ++i)
        {
            SolR::gKernel->render_begin(0);
            SolR::gKernel->render_end();
        }
        SolR::gKernel->resetFrame();
        gLighting = false;
    }
}

void SolR::glTranslatef(GLfloat x, GLfloat y, GLfloat z)
{
    // SolR::gKernel->translate(x*gScale,y*gScale,-z*gScale); // Z is
    // inverted!!
}

void SolR::glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
{
    Vertex angles = {angle * x * PI / 180.f, angle * y * PI / 180.f, angle * z * PI / 180.f};
    SolR::gKernel->rotate(angles.x, angles.y, angles.z);
}

void SolR::gluLookAt(GLdouble eyeX, GLdouble eyeY, GLdouble eyeZ, GLdouble centerX, GLdouble centerY,
                          GLdouble centerZ, GLdouble upX, GLdouble upY, GLdouble upZ)
{
    gEye.x = static_cast<float>(eyeX * gScale);
    gEye.y = static_cast<float>(eyeY * gScale);
    gEye.z = static_cast<float>(eyeZ * gScale - 5000.f);
    gDir.x = static_cast<float>(centerX * gScale);
    gDir.y = static_cast<float>(centerY * gScale);
    gDir.z = static_cast<float>(centerZ * gScale);
}

void SolR::glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height,
                                GLenum format, GLenum type, const GLvoid *data)
{
    ::glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, data);
}

void *SolR::gluNewNurbsRenderer()
{
    return 0;
}

void SolR::glutSpecialFunc(void (*func)(int key, int x, int y))
{
    ::glutSpecialFunc(func);
}

void SolR::glutReshapeFunc(void (*func)(int width, int height))
{
    ::glutReshapeFunc(func);
}

void SolR::glutIdleFunc(void (*func)(void))
{
    ::glutIdleFunc(func);
}

void SolR::gluPerspective(GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar)
{
    gEye.z = -20.f * static_cast<float>(aspect) / tanf(static_cast<float>(fovy)) * gScale;
    gDir.z = static_cast<float>(gEye.z + zNear * gScale + 5000.f);
    gSceneInfo.viewDistance = static_cast<float>(gDir.z + zFar * gScale + 5000.f);
}

void SolR::glutSetCursor(int cursor)
{
    ::glutSetCursor(cursor);
}

void SolR::glPointSize(GLfloat size)
{
    SolR::gKernel->setPointSize(size);
}

void SolR::glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLvoid *data)
{
    ::glReadPixels(x, y, width, height, format, type, data);
}
