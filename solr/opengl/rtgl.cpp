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

GPUKernel *RayTracer::gKernel = 0;

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
void RayTracer::setOpenCLPlatform(const int platform) {
  RayTracer::gOpenCLPlatform = platform;
}

void RayTracer::setOpenCLDevice(const int device) {
  RayTracer::gOpenCLDevice = device;
}
#endif // USE_OPENCL

void RayTracer::InitializeRaytracer(const int width, const int height) {
#ifdef USE_OPENCL
  LOG_INFO(1, "Intializing Raytracing engine: " << gOpenCLPlatform << ","
                                                << gOpenCLDevice);
#else
  LOG_INFO(1, "Intializing Raytracing engine");
#endif // USE_OPENCL
  if (!gKernel) {
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

void RayTracer::glCompactBoxes() {
  if (RayTracer::gKernel) {
    RayTracer::gKernel->compactBoxes(true);
  }
}

/*
________________________________________________________________________________

Create Random Materials
________________________________________________________________________________
*/
void RayTracer::createRandomMaterials(bool update, bool lightsOnly) {
  srand(static_cast<int>(time(0)));
  int start(0);
  int end(NB_MAX_MATERIALS);
  if (lightsOnly) {
    start = 120;
    end = 130;
  }
  // Materials
  long R = 0;
  long G = 0;
  long B = 0;
  for (int i(start); i < end; ++i) {
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

    switch (i) {
    case DEFAULT_LIGHT_MATERIAL:
      r = 1.f;
      g = 1.f;
      b = 1.f;
      innerIllumination.x = 1.f;
      break;
    }

    int material = update ? i : RayTracer::gKernel->addMaterial();
    RayTracer::gKernel->setMaterial(
        material, r, g, b, noise, reflection, refraction, procedural, wireframe,
        wireframeDepth, transparency, opacity, textureId, TEXTURE_NONE,
        TEXTURE_NONE, TEXTURE_NONE, TEXTURE_NONE, TEXTURE_NONE, TEXTURE_NONE,
        specular.x, specular.y, specular.w, innerIllumination.x,
        innerIllumination.y, innerIllumination.z, fastTransparency);
  }
}

void RayTracer::glBegin(GLint mode) { RayTracer::gKernel->setGLMode(mode); }

int RayTracer::glEnd() { return RayTracer::gKernel->setGLMode(-1); }

void RayTracer::glEnable(GLenum cap) {
  switch (cap) {
  case GL_LIGHTING: {
    if (!gLighting) {
      int p = RayTracer::gKernel->addPrimitive(ptSphere);
      RayTracer::gKernel->setPrimitive(
          p, 20.f * gScale, 20.f * gScale, -20.f * gScale, 0.1f * gScale,
          0.1f * gScale, 0.1f * gScale, DEFAULT_LIGHT_MATERIAL);
      gLighting = true;
      LOG_INFO(3, "[OpenGL] Light Added");
    }
  } break;
  }
  //::glEnable(cap);
}

void RayTracer::glDisable(GLenum cap) { ::glDisable(cap); }

void RayTracer::glClear(GLbitfield mask) { ::glClear(mask); }

void RayTracer::glVertex3f(GLfloat x, GLfloat y, GLfloat z) {
  //::glVertex3f(x,y,z);
  RayTracer::gKernel->addVertex(x * gScale, y * gScale, z * gScale);
}

void RayTracer::glVertex3fv(const GLfloat *v) {
  //::glVertex3f(x,y,z);
  RayTracer::gKernel->addVertex(v[0] * gScale, v[1] * gScale, v[2] * gScale);
}

void RayTracer::glNormal3f(GLfloat x, GLfloat y, GLfloat z) {
  //::glNormal3f(x,y,z);
  RayTracer::gKernel->addNormal(x, y, z);
}

void RayTracer::glNormal3fv(const GLfloat *n) {
  RayTracer::gKernel->addNormal(n[0], n[1], n[2]);
}

void RayTracer::glColor3f(GLfloat red, GLfloat green, GLfloat blue) {
  glColor4f(red, green, blue, 0.f);
}

void RayTracer::glColor4f(GLfloat red, GLfloat green, GLfloat blue,
                          GLfloat alpha) {
  /*
  bool found(false);
  unsigned int i(0);
  while( i<RayTracer::gKernel->getNbActiveMaterials() && !found )
  {
     Material* mat=RayTracer::gKernel->getMaterial(i);
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
     RayTracer::gKernel->setMaterialColor(i,red,green,blue);
  }
  else
  */
  {
    int m = RayTracer::gKernel->getCurrentMaterial();
    ++m;
    RayTracer::gKernel->setMaterial(
        m, red, green, blue, 0.f, 0.f, 1.2f, false, 0.f, 0, alpha,
        gSceneInfo.viewDistance, MATERIAL_NONE, MATERIAL_NONE, MATERIAL_NONE,
        MATERIAL_NONE, TEXTURE_NONE, TEXTURE_NONE, TEXTURE_NONE, 1.f, 200.f,
        1000.f, 0.f, 0.f, 0.f, false);
    RayTracer::gKernel->setCurrentMaterial(m);
  }
}

void RayTracer::glRasterPos2f(GLfloat x, GLfloat y) { ::glRasterPos2f(x, y); }

void RayTracer::glRasterPos3f(GLfloat x, GLfloat y, GLfloat z) {
  ::glRasterPos3f(x, y, z);
}

void RayTracer::glTexParameterf(GLenum target, GLenum pname, GLfloat param) {
  ::glTexParameterf(target, pname, param);
}

void RayTracer::glTexCoord2f(GLfloat s, GLfloat t) {
  //::glTexCoord2f(s,t);
  RayTracer::gKernel->addTextCoord(s, t, 0.f);
}

void RayTracer::glTexCoord3f(GLfloat x, GLfloat y, GLfloat z) {
  //::glTexCoord3f(x,y,z);
  RayTracer::gKernel->addTextCoord(x, y, z);
}

void RayTracer::glTexEnvf(GLenum target, GLenum pname, GLfloat param) {
  ::glTexEnvf(target, pname, param);
}

void RayTracer::glTexImage2D(GLenum target, GLint level, GLint internalformat,
                             GLsizei width, GLsizei height, GLint border,
                             GLenum format, GLenum type, const GLvoid *pixels) {
  ::glTexImage2D(target, level, internalformat, width, height, border, format,
                 type, pixels);
}

/*
* GLUT
*/

void RayTracer::glutInit(int *pargc, char **argv) { ::glutInit(pargc, argv); }

void RayTracer::glutInitWindowPosition(int x, int y) {
  ::glutInitWindowPosition(x, y);
}

void RayTracer::glutReshapeWindow(int width, int height) {
  ::glutReshapeWindow(width, height);
}

void RayTracer::glViewport(int a, int b, int width, int height) {
  // Initialize Raytracer
  InitializeRaytracer(width, height);
  ::glViewport(a, b, width, height);
}

void RayTracer::glutInitWindowSize(int width, int height) {
  // Initialize Raytracer
  InitializeRaytracer(width, height);

  ::glutInitWindowSize(width, height);
}

void RayTracer::glutInitDisplayMode(unsigned int displayMode) {
  ::glutInitDisplayMode(displayMode);
}

void RayTracer::glutMainLoop(void) { ::glutMainLoop(); }

int RayTracer::glutCreateWindow(const char *title) {
  return ::glutCreateWindow(title);
}

void RayTracer::glutDestroyWindow(int window) {
  ::glutDestroyWindow(window);
  delete gKernel;
  gKernel = 0;
}

void RayTracer::glutFullScreen(void) { ::glutFullScreen(); }

void RayTracer::glLoadIdentity() {
  RayTracer::render();
  ::glLoadIdentity();
}

int RayTracer::glutGet(GLenum query) { return ::glutGet(query); }

int RayTracer::glutDeviceGet(GLenum query) { return ::glutDeviceGet(query); }

int RayTracer::glutGetModifiers(void) { return ::glutGetModifiers(); }

int RayTracer::glutLayerGet(GLenum query) { return ::glutLayerGet(query); }

void RayTracer::glutKeyboardFunc(void (*callback)(unsigned char, int, int)) {
  ::glutKeyboardFunc(callback);
}

void RayTracer::glutDisplayFunc(void (*callback)(void)) {
  ::glutDisplayFunc(callback);
}

void RayTracer::glutMouseFunc(void (*callback)(int, int, int, int)) {
  ::glutMouseFunc(callback);
}

void RayTracer::glutMotionFunc(void (*callback)(int, int)) {
  ::glutMotionFunc(callback);
}

void RayTracer::glutTimerFunc(unsigned int time, void (*callback)(int),
                              int value) {
  ::glutTimerFunc(time, callback, value);
}

int RayTracer::glutCreateMenu(void (*callback)(int menu)) {
  return ::glutCreateMenu(callback);
}

void RayTracer::glutDestroyMenu(int menu) { ::glutDestroyMenu(menu); }

void RayTracer::glutAddMenuEntry(const char *label, int value) {
  ::glutAddMenuEntry(label, value);
}

void RayTracer::glutAttachMenu(int button) { ::glutAttachMenu(button); }

void RayTracer::glutBitmapString(void *font, const unsigned char *string) {
#ifndef __APPLE__
  ::glutBitmapString(font, string);
#endif
}

void RayTracer::glutPostRedisplay(void) { ::glutPostRedisplay(); }

void RayTracer::glutSwapBuffers(void) { ::glutSwapBuffers(); }

void RayTracer::gluSphere(void *, GLfloat radius, GLint, GLint) {
  int p = RayTracer::gKernel->addPrimitive(ptSphere);
  Vertex translation = RayTracer::gKernel->getTranslation();
  int m = RayTracer::gKernel->getCurrentMaterial();
  RayTracer::gKernel->setPrimitive(
      p, translation.x * gScale, translation.y * gScale, translation.z * gScale,
      radius * gScale, 0.f, 0.f, m);
}

void RayTracer::glutWireSphere(GLdouble radius, GLint, GLint) {
  int p = RayTracer::gKernel->addPrimitive(ptSphere);
  Vertex translation = RayTracer::gKernel->getTranslation();
  int m = RayTracer::gKernel->getCurrentMaterial();
  RayTracer::gKernel->setPrimitive(
      p, translation.x * gScale, translation.y * gScale, translation.z * gScale,
      static_cast<float>(radius) * gScale, 0.f, 0.f, m);
}

GLUquadricObj *RayTracer::gluNewQuadric(void) { return 0; }

void RayTracer::glClearColor(GLfloat red, GLfloat green, GLfloat blue,
                             GLfloat alpha) {
  gSceneInfo.backgroundColor.x = red;
  gSceneInfo.backgroundColor.y = green;
  gSceneInfo.backgroundColor.z = blue;
  gSceneInfo.backgroundColor.w = alpha;
}

void RayTracer::glMaterialfv(GLenum face, GLenum pname, const GLfloat *params) {
  switch (pname) {
  case GL_AMBIENT:
    RayTracer::glColor3f(params[0], params[1], params[2]);
    break;
  }
}

void RayTracer::glGenTextures(GLsizei n, GLuint *textures) {
  ++gCurrentTexture;
  RayTracer::gKernel->setTexturesTransfered(false);
  *textures = gCurrentTexture;
}

void RayTracer::glBindTexture(GLenum target, GLuint texture) {
  switch (target) {
  case GL_TEXTURE_2D:
    RayTracer::gKernel->setMaterialTextureId(gCurrentTexture);
    break;
  }
}

int RayTracer::gluBuild2DMipmaps(GLenum target, GLint components, GLint width,
                                 GLint height, GLenum format, GLenum type,
                                 const void *data) {
  TextureInformation textureInfo;
  textureInfo.size.x = width;
  textureInfo.size.y = height;

  switch (format) {
  case GL_RGB:
    textureInfo.size.z = 3;
    break;
  case GL_RGBA:
    textureInfo.size.z = 4;
    break;
  }
  textureInfo.offset = 0;
  textureInfo.buffer = (unsigned char *)data;

  RayTracer::gKernel->setTexture(gCurrentTexture, textureInfo);
  return 0;
}

void RayTracer::setAngles(GLfloat x, GLfloat y, GLfloat z) {
  gAngles.x = x;
  gAngles.y = y;
  gAngles.z = z;
}

void RayTracer::glFlush() { ::glFlush(); }

void RayTracer::glTexSubImage2D(GLenum target, GLint level, GLint xoffset,
                                GLint yoffset, GLsizei width, GLsizei height,
                                GLenum format, GLenum type, const GLvoid *data);

void RayTracer::glPushAttrib(GLbitfield mask) {}

void RayTracer::glPopAttrib() {}

void RayTracer::glTexParameteri(GLenum target, GLenum pname, GLint param) {}

void RayTracer::glBlendFunc(GLenum sfactor, GLenum dfactor) {}

void RayTracer::glMatrixMode(GLenum mode) {}

void RayTracer::glPushMatrix() {}

void RayTracer::glPopMatrix() {}

GLenum RayTracer::glGetError() { return ::glGetError(); }

void RayTracer::glVertex2i(GLint x, GLint y) {
  glVertex3f(static_cast<GLfloat>(x), static_cast<GLfloat>(y), 0.f);
}

void RayTracer::glOrtho(GLdouble left, GLdouble right, GLdouble bottom,
                        GLdouble top, GLdouble nearVal, GLdouble farVal) {
  // gMisc.w = 1;
}

void RayTracer::render() {
  if (RayTracer::gKernel) {
    Vertex rotation = RayTracer::gKernel->getRotation();
    RayTracer::gKernel->setCamera(gEye, gDir, rotation);
    RayTracer::gKernel->setSceneInfo(gSceneInfo);
    RayTracer::gKernel->setPostProcessingInfo(glPostProcessingInfo);
    if (!gLighting) {
      // if no light is defined, I add one
      int p = RayTracer::gKernel->addPrimitive(ptSphere);
      RayTracer::gKernel->setPrimitive(
          p, 20.f * gScale, 20.f * gScale, 20.f * gScale, 0.1f * gScale,
          0.1f * gScale, 0.1f * gScale, DEFAULT_LIGHT_MATERIAL);

      // p = RayTracer::gKernel->addPrimitive(ptSphere);
      // RayTracer::gKernel->setPrimitive(p,0.f,0.f,0.f,0.1f*gScale,0.1f*gScale,0.1f*gScale,0);
      gLighting = true;
    }

    RayTracer::gKernel->compactBoxes(false);
    for (int i(0); i < gTotalPathTracingIterations; ++i) {
      RayTracer::gKernel->render_begin(0);
      RayTracer::gKernel->render_end();
    }
    RayTracer::gKernel->resetFrame();
    gLighting = false;
  }
}

void RayTracer::glTranslatef(GLfloat x, GLfloat y, GLfloat z) {
  // RayTracer::gKernel->translate(x*gScale,y*gScale,-z*gScale); // Z is
  // inverted!!
}

void RayTracer::glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z) {
  Vertex angles = {angle * x * PI / 180.f, angle * y * PI / 180.f,
                   angle * z * PI / 180.f};
  RayTracer::gKernel->rotate(angles.x, angles.y, angles.z);
}

void RayTracer::gluLookAt(GLdouble eyeX, GLdouble eyeY, GLdouble eyeZ,
                          GLdouble centerX, GLdouble centerY, GLdouble centerZ,
                          GLdouble upX, GLdouble upY, GLdouble upZ) {
  gEye.x = static_cast<float>(eyeX * gScale);
  gEye.y = static_cast<float>(eyeY * gScale);
  gEye.z = static_cast<float>(eyeZ * gScale - 5000.f);
  gDir.x = static_cast<float>(centerX * gScale);
  gDir.y = static_cast<float>(centerY * gScale);
  gDir.z = static_cast<float>(centerZ * gScale);
}

void RayTracer::glTexSubImage2D(GLenum target, GLint level, GLint xoffset,
                                GLint yoffset, GLsizei width, GLsizei height,
                                GLenum format, GLenum type,
                                const GLvoid *data) {
  ::glTexSubImage2D(target, level, xoffset, yoffset, width, height, format,
                    type, data);
}

void *RayTracer::gluNewNurbsRenderer() { return 0; }

void RayTracer::glutSpecialFunc(void (*func)(int key, int x, int y)) {
  ::glutSpecialFunc(func);
}

void RayTracer::glutReshapeFunc(void (*func)(int width, int height)) {
  ::glutReshapeFunc(func);
}

void RayTracer::glutIdleFunc(void (*func)(void)) { ::glutIdleFunc(func); }

void RayTracer::gluPerspective(GLdouble fovy, GLdouble aspect, GLdouble zNear,
                               GLdouble zFar) {
  gEye.z = -20.f * static_cast<float>(aspect) / tanf(static_cast<float>(fovy)) *
           gScale;
  gDir.z = static_cast<float>(gEye.z + zNear * gScale + 5000.f);
  gSceneInfo.viewDistance = static_cast<float>(gDir.z + zFar * gScale + 5000.f);
}

void RayTracer::glutSetCursor(int cursor) { ::glutSetCursor(cursor); }

void RayTracer::glPointSize(GLfloat size) {
  RayTracer::gKernel->setPointSize(size);
}

void RAYTRACINGENGINE_API RayTracer::glReadPixels(GLint x, GLint y,
                                                  GLsizei width, GLsizei height,
                                                  GLenum format, GLenum type,
                                                  GLvoid *data) {
  ::glReadPixels(x, y, width, height, format, type, data);
}
