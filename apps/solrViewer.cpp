/*
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#define _CRT_SECURE_NO_WARNINGS

// OpenGL Graphics Includes
#include "../solr/Logging.h"
#include "../solr/opengl/rtgl.h"
using namespace RayTracer;

#define _USE_MATH_DEFINES

// Includes
#include <cassert>
#include <iostream>
#include <locale>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>

// Raytracer
#include "../solr/Consts.h"
#include "../solr/images/jpge.h"

// Project
#ifdef USE_KINECT
#include "KinectFaceTrackingScene.h"
#endif // USE_KINECT
#include "scenes/AnimationScene.h"
#include "scenes/CausticScene.h"
#include "scenes/CausticScene.h"
#include "scenes/CornellBoxScene.h"
#include "scenes/CubesScene.h"
#include "scenes/CylinderScene.h"
#include "scenes/DoggyStyleScene.h"
#include "scenes/FractalScene.h"
#include "scenes/FractalsScene.h"
#include "scenes/GalleryScene.h"
#include "scenes/GraphScene.h"
#include "scenes/MetaballsScene.h"
#include "scenes/MoleculeScene.h"
#include "scenes/ObjScene.h"
#include "scenes/PerpetualMotionScene.h"
#include "scenes/SpindizzyScene.h"
#include "scenes/SwcScene.h"
#include "scenes/TransparentScene.h"
#include "scenes/TrefoilKnotScene.h"
#include "scenes/TrianglesScene.h"
#include "scenes/WaterScene.h"
#include "scenes/XmasScene.h"
#include "scenes/Year2013.h"

// General Settings
const int TARGET_FPS = 200;
const int REFRESH_DELAY = 1000 / TARGET_FPS; // ms

// ----------------------------------------------------------------------
// Scene
// ----------------------------------------------------------------------
bool gCopyright(false);
int gTest(0);
bool gAutoFocus(false);
int gCornellBoxType(0);
bool gAnimate(false);
FLOAT4 gBkColor = {0.f, 0.f, 0.f, 0.f};
bool gDraft(false);
int gSphereMaterial = 0;
int gGroundMaterial = 0;
bool gSuspended(false);
bool gSavedToDisk(false);
std::string gHint;

#ifdef WIN32
ULONGLONG gRenderingTime;
#endif

// ----------------------------------------------------------------------
// Benchmark
// ----------------------------------------------------------------------
#ifdef WIN32
int gTickCount = 0;
#endif // WIN32
bool gBenchmarking(false);
int gOptimalNbOfBoxes(12000);
int gTreeDepth(9);

// Oculus
float gDistortion = 0.1f;

// GPU
int gDrawing(false);

// Command line arguments
std::string gFilename;

// Rendering window vars
#ifdef USE_OCULUS
unsigned int gWindowWidth = 1280 / 2;
unsigned int gWindowHeight = 800 / 2;
#else
unsigned int gWindowWidth = 800;
unsigned int gWindowHeight =
    static_cast<unsigned int>(gWindowWidth * 9.f / 16.f);
#endif // USE_OCULUS
unsigned int gWindowDepth = 4;

// Scene
Scene *gScene = 0;

// Menu
struct MenuItem {
  std::string description;
  char key;
};
const int NB_MENU_ITEMS = 20;
MenuItem menuItems[NB_MENU_ITEMS] = {
    {"a: Black background and no image noise", 'a'},
    {"b: Randomly set background color and image noise", 'b'},
    {"f: Full screen", 'f'},
    {"h: Help", 'h'},
    {"i: Bounding boxes", 'i'},
    {"m: Animate scene", 'm'},
    {"n: Next 3D model (in folder ./obj)", 'n'},
    {"o: Perpective/Isometric 3D/Antialiasing", 'o'},
    {"p: Post processing (Depth of field, ambient occlusion, enlightment", 'p'},
    {"r: Reset current scene", 'r'},
    {"t: Next scene", 't'},
    {"s: Change graphics level", 's'},
    {"v: Random materials", 'i'},
    {"x: Set environment (CornellBox, SkyBox, etc.)", 'x'},
    {"*: View modes (Standard, Anaglyth 3D, Oculus Rift", '*'},
    {"1: Camera mouse control", '1'},
    {"2: Light mouse control", '2'},
    {"3: 3D model mouse control", '3'},
    {"9: Shadow strength", '9'},
    {"Escape: Exit application", '\033'}};

// controlType
enum ControlType {
  ctCamera = 0,
  ctLightSource = 1,
  ctObject = 2,
  ctFocus = 3,
  ctFOV = 4,
  ct3DVision = 5,
  ctGroundHeight = 6
};
ControlType gControlType = ctCamera;
int gSelectedPrimitive = -1;

// Materials
int currentMaterial = 0;
int gMaterials[2];
int gWallTexture = 3;

// Scene description and behavior
int gNbBoxes = 0;
int m_nbPrimitives = 0;
int gLampId = 0;
int gNbMaterials = 0;
int gNbTextureTiles = 1;

Vertex gRotationAngles = {0.f, 0.f, 0.f};
Vertex gViewAngles = {0.f, 0.f, 0.f, 6400.f};
Vertex gViewPos = {0.f, 0.f, -15000.f, 0.f};
Vertex gViewDir = {0.f, 0.f, 0.f, 0.f};
PostProcessingInfo gPostProcessingInfo = {ppe_none, 0.f, 10.f, 1000};

float groundHeight = -3000.f;

// materials
float gReflection = 0.f;
float gRefraction = 1.33f; // Water
float gTransparency = 0.9f;
float gSpecValue = 1.f;
float gSpecPower = 100.f;
float gSpecCoef = 1.f;
float gInnerIllumination = 0.f;
bool gNoise(false);

// --------------------------------------------------------------------------------
// OpenGL
// --------------------------------------------------------------------------------
int gTimebase(0);
int gFrame(0);
int gFPS(0);

bool gHelp(false);

// GL functionality
void initgl(int argc, char **argv);
void display();
void idle();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
void reshape(int x, int y);
void createScene();
void RenderString(float x, float y, float z, void *font,
                  const std::string &string, const Vertex &rgb);

// Helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Sim and Auto-Verification parameters
float anim = 0.f;
bool bNoPrompt = false;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;

unsigned int *uiOutput = 0;

void readParameters(const std::vector<std::string> &parameters) {
  std::vector<std::string>::const_iterator it = parameters.begin();
  while (it != parameters.end()) {
    std::string argument(*it);
    size_t eqPos = argument.find("=");
    if (eqPos != std::string::npos) {
      std::string key(argument.substr(0, eqPos));
      std::string value(argument.substr(eqPos + 1));
      LOG_INFO(1, "Argument: " << key << " = " << value);
#ifdef USE_OPENCL
      if (key.find("-platform") != std::string::npos)
        RayTracer::setOpenCLPlatform(atoi(value.c_str()));
      if (key.find("-device") != std::string::npos)
        RayTracer::setOpenCLDevice(atoi(value.c_str()));
#endif // USE_OPENCL
      if (key.find("-objFile") != std::string::npos)
        gFilename = value.c_str();
      if (key.find("-width") != std::string::npos)
        gWindowWidth = atoi(value.c_str());
      if (key.find("-height") != std::string::npos)
        gWindowHeight = atoi(value.c_str());
      if (key.find("-benchmark") != std::string::npos)
        gBenchmarking = (atoi(value.c_str()) == 1);
      if (key.find("-scene") != std::string::npos) {
        gTest = atoi(value.c_str());
        LOG_INFO(1, "Scene: " << gTest);
      }
      if (key.find("-cornellBox") != std::string::npos)
        gCornellBoxType = atoi(value.c_str());
    }
    ++it;
  }
}

/*
 ________________________________________________________________________________

 idle
 ________________________________________________________________________________
 */
void idle() {}

/*
 ________________________________________________________________________________

 cleanup
 ________________________________________________________________________________
 */
void cleanup() {}

/*
 ________________________________________________________________________________

 Setup the window and assign callbacks
 ________________________________________________________________________________
 */
void initgl(int argc, char **argv) {
  glutInit(&argc, (char **)argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowPosition(glutGet(GLUT_SCREEN_WIDTH) / 2 - gWindowWidth / 2,
                         glutGet(GLUT_SCREEN_HEIGHT) / 2 - gWindowHeight / 2);
  std::string caption("SoL-R tech demo v00.02.00");
#ifdef USE_CUDA
  caption += " (Powered by CUDA)";
#endif
#ifdef USE_OPENCL
  caption += " (Powered by OpenCL)";
#endif
  glutInitWindowSize(gWindowWidth, gWindowHeight);
  glutCreateWindow(caption.c_str());
  glutDisplayFunc(display); // register GLUT callback functions
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutIdleFunc(idle);
  glutReshapeFunc(reshape);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 1);
}

/*
 ________________________________________________________________________________

 Render string to OpenGL
 ________________________________________________________________________________
 */
void RenderString(float x, float y, float z, void *font,
                  const std::string &string, const Vertex &rgb) {
  // glColor3f(rgb.x, rgb.y, rgb.z);
  glRasterPos3f(x, y, z);
  glutBitmapString(font,
                   reinterpret_cast<const unsigned char *>(string.c_str()));
}

/*
 ________________________________________________________________________________

 Display callback
 ________________________________________________________________________________
 */
void display() {
  if (gSuspended)
    return;
  // clear graphics
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  Vertex textColor = {0.f, 0.f, 0.f};
  SceneInfo &si = gScene->getSceneInfo();

  gFrame++;
  int time = glutGet(GLUT_ELAPSED_TIME);
  int processingTime = time - gTimebase;
  if (processingTime > 1000) {
    gFPS = gFrame * 1000 / processingTime;
    gTimebase = time;
    gFrame = 0;
  }

  if (!gDrawing) {
    gDrawing = true;

    SceneInfo &si = gScene->getSceneInfo();
    if (si.pathTracingIteration == 0) {
#ifdef WIN32
      gRenderingTime = GetTickCount64();
#endif
      gSavedToDisk = false;
    };
#ifdef WIN32
    if (si.pathTracingIteration == si.maxPathTracingIterations - 1) {
      gRenderingTime = GetTickCount64() - gRenderingTime;
    }
#endif

    if (si.pathTracingIteration < si.maxPathTracingIterations) {
      gScene->getKernel()->setCamera(gViewPos, gViewDir, gViewAngles);
      gScene->getKernel()->setPostProcessingInfo(gPostProcessingInfo);
      gScene->render(gAnimate);
      if (gAnimate) {
        si.pathTracingIteration = 0;
      } else {
        si.pathTracingIteration++;
      }
#ifdef WIN32
      if (gHelp) {
        char tmp[1024];
        std::string tmpMenuItems;
        for (int i(0); i < NB_MENU_ITEMS; ++i) {
          tmpMenuItems += menuItems[i].description;
          tmpMenuItems += "\n";
        }
        Vertex textColor = {1.f, 1.f, 1.f};
        sprintf(tmp, "%sSelected primitive: %d\nFPS: %d on %s (%dx%d)\nScene "
                     "%d: %s\nMouse control: %s\nPrimitives/Boxes: "
                     "%d/%d\nhttp://cudaopencl.blogspot.com [%d]",
                tmpMenuItems.c_str(), gSelectedPrimitive, gFPS,
                RayTracer::gKernel->getGPUDescription().c_str(),
                gScene->getSceneInfo().size.x, gScene->getSceneInfo().size.y,
                gTest, gScene->getName().c_str(), gHint.c_str(),
                RayTracer::gKernel->getNbActivePrimitives(),
                RayTracer::gKernel->getNbActiveBoxes(),
                si.pathTracingIteration);
        RenderString(-0.9f, 0.9f, 0.f, GLUT_BITMAP_HELVETICA_10, tmp,
                     textColor);
      }
#endif // WIN32
    } else {
      gScene->getKernel()->render_end();

      if (gBenchmarking)
        Cleanup(EXIT_SUCCESS);
    }

    // Screenshot
    if (si.pathTracingIteration == si.maxPathTracingIterations) {
#ifdef WIN32
      // Rendering time
      std::stringstream text;
      text << "Rendered in ";
      text << gRenderingTime << " ms\n";
      text << gKernel->getGPUDescription();
      text << "\n"
           << gKernel->getNbActivePrimitives() << " triangles, " << si.size.x
           << "x" << si.size.y << ", " << si.maxPathTracingIterations
           << " samples/pixel";
      RenderString(-0.95f, -0.8f, 0.f, GLUT_BITMAP_HELVETICA_10,
                   text.str().c_str(), textColor);
      gScene->renderText();
#endif

      if (!gSavedToDisk) {
        int margin = 32;
        size_t size = (si.size.x - margin) * (si.size.y - margin) * gColorDepth;
        GLubyte *buffer = new GLubyte[size];
        glReadPixels(0, 0, si.size.x - margin, si.size.y - margin, GL_RGB,
                     GL_UNSIGNED_BYTE, buffer);
        GLubyte *dst = new GLubyte[size];
        int X = (si.size.x - margin) * gColorDepth;

        // Vertical flip
        for (int y(0); y < (si.size.y - margin); ++y) {
          int idxSrc = y * X;
          memcpy(dst + idxSrc, buffer + (size - idxSrc), X);
        }

        // Save to disc
        std::string filename("./raytracer");
        /*
         time_t rawtime;
         struct tm * timeinfo;
         char dateAndTime[80];
         std::time(&rawtime);
         timeinfo = localtime(&rawtime);
         strftime(dateAndTime,80,"_%Y-%m-%d_%H-%M-%S",timeinfo);
         filename+=dateAndTime;
         */
        filename += ".jpg";
        jpge::compress_image_to_jpeg_file(filename.c_str(), si.size.x - margin,
                                          si.size.y - margin, gColorDepth, dst);
        delete[] buffer;
        delete[] dst;
        gSavedToDisk = true;
      }
    }

// TODO: Copyright
#ifdef WIN32
    if (gCopyright) {
      // Copyright
      const char *copyright = "http://cudaopencl.blogspot.com";
      float p = strlen(copyright) * 20.f / si.size.x;
      RenderString(-p / 2.f, 0.f, 0.f, GLUT_BITMAP_HELVETICA_18, copyright,
                   textColor);
      gScene->renderText();
    }

#endif
    glutSwapBuffers();
    gDrawing = false;
  }
}

/*
 ________________________________________________________________________________

 Timer Event
 ________________________________________________________________________________
 */
void timerEvent(int value) {
#ifdef _USE_KINECT
  animateSkeleton();
#endif // USE_KINECT
#ifdef _USE_KINECT
case 10: // Kinect
{
  int p = 0;
  int b = 0;
  char *depthBitmap = kernel.getDepthBitmap();
  if (depthBitmap != 0) {
    for (int y(0); y < KINECT_DEPTH_HEIGHT; y += gKinectStep) {
      for (int x(0); x < KINECT_DEPTH_WIDTH; x += gKinectStep) {
        int index = KINECT_DEPTH_SIZE -
                    (y * KINECT_DEPTH_WIDTH + x) * KINECT_DEPTH_DEPTH;
        char a = depthBitmap[index];
        char b = depthBitmap[index + 1];
        int s = b * 256 + a;

        // USHORT realDepth = (s & 0xfff8) >> 3;

        USHORT player = s & 7;
        /*
         if( player == 0 )
         {
         kernel.setPrimitive( p, 5+p/gKinectNbSpherePerBox,
         0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
         0, 1, 1 );
         }
         else
         {
         kernel.setPrimitive( p, 5+p/gKinectNbSpherePerBox,
         (x-160)*gKinectSpace, (y-120)*gKinectSpace, s/5.f, gKinectSize, 0.f,
         0.f,
         20+x%20, 1, 1 );
         }
         */

        if (s < 1 || s > 15000)
          s = 50000;
        // if( player == 0 ) s = 50000;

        kernel.setPrimitive(p, 5 + p / gKinectNbSpherePerBox,
                            -(x - 160) * gKinectSpace, (y - 120) * gKinectSpace,
                            s / 3.f - 5000.f, gKinectSize, 0.f, 0.f, 7, 1, 1);

        ++b;
        ++p;
      }
    }
  }
  kernel.compactBoxes();
  break;
}
#endif // USE_KINECT
  glutPostRedisplay();
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void reshape(int x, int y) {
  if (x % 32 != 0)
    x += 32 - (x % 32);
  if (y % 32 != 0)
    y += 32 - (y % 32);
  // y = x*9/16;
  SceneInfo &sceneInfo = gScene->getSceneInfo();
  sceneInfo.size.x = x;
  sceneInfo.size.y = y;
  sceneInfo.pathTracingIteration = 0;
  gScene->getKernel()->reshape();
  glViewport(0, 0, x, y);
  gWindowWidth = x;
  gWindowHeight = y;
  // glutPostRedisplay();
}

/*
 ________________________________________________________________________________

 Menu
 ________________________________________________________________________________
 */
void mainMenu(int i) { keyboard((unsigned char)i, 0, 0); }

void initMenus() {
  glutCreateMenu(mainMenu);
  for (int i(0); i < NB_MENU_ITEMS; ++i) {
    glutAddMenuEntry(menuItems[i].description.c_str(), menuItems[i].key);
  }
  glutAttachMenu(GLUT_MIDDLE_BUTTON);
}

/*
 ________________________________________________________________________________

 Keyboard events handler
 ________________________________________________________________________________
 */
void keyboard(unsigned char key, int x, int y) {
  float step(100.f);
  GPUKernel *kernel = gScene->getKernel();
  switch (key) {
  case 'q':
  case 'Q': {
    GPUKernel *kernel = gScene->getKernel();
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[255];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, 255, "%Y-%m-%d_%H-%M-%S.jpg", timeinfo);
    if (key == 'q') {
#ifdef WIN32
      std::string filename(
          "E:/Cloud/Dropbox/Samsung Link/Photos/1K/CudaRayTracer_");
#else
      std::string filename("~/Pictures/1K/CudaRayTracer_");
#endif
      filename += buffer;

      if (kernel)
        kernel->generateScreenshot(
            filename, 2100, 2970,
            gScene->getSceneInfo().maxPathTracingIterations);
      // kernel->generateScreenshot(filename, 2970, 2100,
      // gScene->getSceneInfo().maxPathTracingIterations);
    } else {
#ifdef WIN32
      std::string filename(
          "E:/Cloud/Dropbox/Samsung Link/Photos/4K/CudaRayTracer_");
#else
      std::string filename("~/Pictures/4K/CudaRayTracer_");
#endif
      filename += buffer;
      if (kernel)
        // kernel->generateScreenshot(filename, 2970, 2100,
        // gScene->getSceneInfo().maxPathTracingIterations);
        // kernel->generateScreenshot(filename, 2 * 2100, 2 * 2970,
        // gScene->getSceneInfo().maxPathTracingIterations);
        kernel->generateScreenshot(
            filename, 2 * 2970, 2 * 2100,
            gScene->getSceneInfo().maxPathTracingIterations);
    }
    break;
  }
  case ' ': {
    // Reset scene
    gSuspended = !gSuspended;
    break;
  }
  case 'n': {
    // Reset scene
    delete gScene;
    gScene = 0;
    m_counter++;
    createScene();
    break;
  }
  case 'S': {
    gScene->getSceneInfo().graphicsLevel++;
    if (gScene->getSceneInfo().graphicsLevel > 4)
      gScene->getSceneInfo().graphicsLevel = 0;
    break;
  }
  case 'r': {
    // Reset scene
    delete gScene;
    gScene = 0;
    createScene();
    break;
  }
  case 'R': {
    // Reset scene
    gScene->getKernel()->recompileKernels(
        "~/git/Sol-R/solr/opencl/RayTracer.cl");
    break;
  }
  case 'F': {
    // Toggle to full screen mode
    glutFullScreen();
    break;
  }
  case 'G': {
    gScene->getSceneInfo().nbRayIterations = NB_MAX_ITERATIONS;
    break;
  }
  case 'g': {
    int nbTextures = (kernel->getNbActiveTextures() - gScene->getNbHDRI()) / 5;
    if (nbTextures > 0) {
      gGroundMaterial = (gGroundMaterial + 1) % nbTextures;
      int index = gScene->getNbHDRI() + 5 * gGroundMaterial;

      Material *m = kernel->getMaterial(SKYBOX_GROUND_MATERIAL);
      int bumpTextureId = MATERIAL_NONE;
      int normalTextureId = MATERIAL_NONE;
      int diffuseTextureId = MATERIAL_NONE;
      int specularTextureId = MATERIAL_NONE;
      int reflectionTextureId = MATERIAL_NONE;
      int transparencyTextureId = MATERIAL_NONE;
      for (int t(0); t < 5; ++t) {
        int idx = index + t;
        TextureInformation ti = kernel->getTextureInformation(idx);
        switch (ti.type) {
        case tex_bump:
          bumpTextureId = idx;
          break;
        case tex_normal:
          normalTextureId = idx;
          break;
        case tex_diffuse:
          diffuseTextureId = idx;
          break;
        case tex_transparent:
          transparencyTextureId = idx;
          break;
        case tex_specular:
          specularTextureId = idx;
          break;
        case tex_reflective:
          reflectionTextureId = idx;
          break;
        }
      }
      if (transparencyTextureId == TEXTURE_NONE) {
        m->transparency = 0.f;
        m->refraction = 0.f;
      } else {
        m->transparency = 1.f;
        m->refraction = 1.1f;
      }
      kernel->setMaterial(
          SKYBOX_GROUND_MATERIAL, m->color.x, m->color.y, m->color.z,
          m->color.w, 1.f /*m->reflection.x*/, m->refraction,
          (m->attributes.y == 1), (m->attributes.z == 1), m->attributes.w,
          m->transparency, m->opacity, diffuseTextureId, normalTextureId,
          bumpTextureId, specularTextureId, reflectionTextureId,
          transparencyTextureId, TEXTURE_NONE, m->specular.x, m->specular.y,
          m->specular.z, m->innerIllumination.x, m->innerIllumination.y,
          m->innerIllumination.z, (m->attributes.x == 1));
    }
    break;
  }
  case 'm': {
    gAnimate = !gAnimate;
    break;
  }
  case 'o': {
    gScene->getSceneInfo().misc.w =
        (gScene->getSceneInfo().misc.w == 3) ? 0 : 3;
    gScene->getSceneInfo().graphicsLevel =
        (gScene->getSceneInfo().misc.w == 3) ? 1 : 4;
    gScene->getSceneInfo().backgroundColor.w =
        (gScene->getSceneInfo().misc.w == 3) ? 0.01f : 0.5f;
    break;
  }
  case 'O': {
    gScene->getKernel()->switchOculusVR();
    break;
  }
  case 'B': {
    gScene->getSceneInfo().backgroundColor.x = 1.f;
    gScene->getSceneInfo().backgroundColor.y = 1.f;
    gScene->getSceneInfo().backgroundColor.z = 1.f;
    gScene->getSceneInfo().backgroundColor.w = 0.5f;
    gScene->getSceneInfo().skybox.y =
        (gScene->getSceneInfo().skybox.y == MATERIAL_NONE)
            ? SKYBOX_SPHERE_MATERIAL
            : MATERIAL_NONE;
    break;
  }
  case 'b': {
    FLOAT4 color = {rand() % 255 / 255.f, rand() % 255 / 255.f,
                    rand() % 255 / 255.f, 0.5f};
    gScene->getSceneInfo().backgroundColor = color;
    break;
  }
  case 'a': {
    // Translate camera position
    gViewPos.x -= 100.f;
    gViewDir.x -= 100.f;
    break;
  }
  case 'd': {
    // Translate camera position
    gViewPos.x += 100.f;
    gViewDir.x += 100.f;
    break;
  }
  case 'w': {
    // Translate camera position
    gViewPos.y += 100.f;
    gViewDir.y += 100.f;
    break;
  }
  case 's': {
    // Translate camera position
    gViewPos.y -= 100.f;
    gViewDir.y -= 100.f;
    break;
  }
  case 'A': {
    gScene->getSceneInfo().backgroundColor = gBkColor;
    break;
  }
  case 'f': {
    gAutoFocus = !gAutoFocus;
    break;
  }
  case 't': {
    // Reset scene
    delete gScene;
    gScene = 0;
    gAnimate = false;
    gTest++;
    gTest = (gTest % 16);
    createScene();
    break;
  }
  case 'T': {
    // Double sided triangles
    int p = gScene->getSceneInfo().parameters.x;
    p = (p == 0) ? 1 : 0;
    gScene->getSceneInfo().parameters.x = p;
    break;
  }
  case 'D': {
    // Draft mode
    gDraft = !gDraft;
    break;
  }
  case 'e': {
    int nbTextures = kernel->getNbActiveTextures();
    int nbHDRI = gScene->getNbHDRI();
    if (nbTextures >= nbHDRI) {
      Material *m = kernel->getMaterial(SKYBOX_SPHERE_MATERIAL);
      gSphereMaterial = (gSphereMaterial + 1) % nbHDRI;
      kernel->setMaterial(SKYBOX_SPHERE_MATERIAL, m->color.x, m->color.y,
                          m->color.z, m->color.w, 0.f /*m->reflection.x*/, 0.f,
                          (m->attributes.y == 1), (m->attributes.z == 1),
                          m->attributes.w, 0.f, m->opacity, gSphereMaterial,
                          TEXTURE_NONE, TEXTURE_NONE, TEXTURE_NONE,
                          TEXTURE_NONE, TEXTURE_NONE, TEXTURE_NONE,
                          m->specular.x, m->specular.y, m->specular.z,
                          m->innerIllumination.x, m->innerIllumination.y,
                          m->innerIllumination.z, (m->attributes.x == 1));
    }
    break;
  }
  case 'E': {
    // Extended geometry
    int p = gScene->getSceneInfo().parameters.y;
    p = (p == 0) ? 1 : 0;
    gScene->getSceneInfo().parameters.y = p;
    break;
  }
  case 'h': {
    gHelp = !gHelp;
    break;
  }
  case 'i': {
    gScene->getSceneInfo().renderBoxes++;
    if (gScene->getSceneInfo().renderBoxes == 2)
      gScene->getSceneInfo().renderBoxes = 0;
    LOG_INFO(1, "Render Boxes: " << gScene->getSceneInfo().renderBoxes);
    break;
  }
  case 'I': {
    gScene->getSceneInfo().parameters.z =
        (gScene->getSceneInfo().parameters.z + 1) % 3;
    break;
  }
  case 'v': {
    gScene->createRandomMaterials(true, false);
    break;
  }
  case 'l': {
    GPUKernel *kernel = gScene->getKernel();
    if (kernel) {
      kernel->saveToFile("test.irt");
    }
    break;
  }
  case 'L': {
    GPUKernel *kernel = gScene->getKernel();
    if (kernel) {
      kernel->resetAll();
      kernel->loadFromFile("test.irt");
      kernel->compactBoxes(true);
    }
    break;
  }
  case 'x': {
    // Reset scene
    delete gScene;
    gCornellBoxType++;
    gCornellBoxType = gCornellBoxType % 10;
    gScene = 0;
    createScene();
    break;
  }
  case 'p': {
    gPostProcessingInfo.type++;
    gPostProcessingInfo.type %= 6;
    break;
  }
  case 'y':
  case 'Y': {
    int light = gScene->getKernel()->getLight(gLampId);
    CPUPrimitive *p = gScene->getKernel()->getPrimitive(light);
    Material *m = gScene->getKernel()->getMaterial(p->materialId);
    if (m) {
      if (key == 'Y') {
        m->color.x = 1.f;
        m->color.y = 1.f;
        m->color.z = 1.f;
      } else {
        m->color.x = rand() % 256 / 256.f;
        m->color.y = rand() % 256 / 256.f;
        m->color.z = rand() % 256 / 256.f;
      }
      gScene->getKernel()->compactBoxes(false);
    }
    break;
  }
  case '*': {
    gScene->getSceneInfo().renderingType =
        (gScene->getSceneInfo().renderingType + 1) % 5;
    gScene->getSceneInfo().graphicsLevel =
        (gScene->getSceneInfo().renderingType == cmVolumeRendering) ? 0 : 4;
    break;
  }
  case '1': {
    gControlType = ctCamera;
    gHint = "Camera";
    break;
  }
  case '2': {
    gControlType = ctLightSource;
    unsigned int activeLamps = RayTracer::gKernel->getNbActiveLamps();

    std::stringstream s;
    s << "Light source (" << gLampId + 1 << "/" << activeLamps << ")";
    LOG_INFO(3, s.str());
    gHint = s.str();
    if (activeLamps > 1) {
      gLampId++;
      gLampId %= activeLamps;
    }
    break;
  }
  case '3': {
    gControlType = ctObject;
    gHint = "Object";
    break;
  }
  case '4': {
    gControlType = ctFocus;
    gHint = "Focus";
    break;
  }
  case '5': {
    gControlType = ctFOV;
    gHint = "Field of view";
    break;
  }
  case '6': {
    gControlType = ctGroundHeight;
    gHint = "Ground Height";
    break;
  }
  case '7': {
    ++gPostProcessingInfo.param3;
    gPostProcessingInfo.param3 %= 6;
    break;
  }
  case '9': {
    gScene->getSceneInfo().shadowIntensity += 0.05f;
    gScene->getSceneInfo().shadowIntensity =
        (gScene->getSceneInfo().shadowIntensity > 1.f)
            ? 0.f
            : gScene->getSceneInfo().shadowIntensity;
    break;
  }
  case '+': {
    if (gControlType == ctLightSource) {
      int light = gScene->getKernel()->getLight(gLampId);
      CPUPrimitive *p = gScene->getKernel()->getPrimitive(light);
      Material *m = gScene->getKernel()->getMaterial(p->materialId);
      if (m) {
        if (m->innerIllumination.x < 1.f) {
          m->innerIllumination.x += 0.01f;
          gScene->getKernel()->compactBoxes(false);
        }
      }
    } else {
      gDistortion += 0.01f;
      RayTracer::gKernel->setDistortion(gDistortion);
      LOG_INFO(1, "Distortion = " << gDistortion);
    }
    break;
  }
  case 'W': {
    gScene->getSceneInfo().backgroundColor.w += 0.1f;
    gScene->getSceneInfo().backgroundColor.w =
        (gScene->getSceneInfo().backgroundColor.w > 1.f)
            ? 0.f
            : gScene->getSceneInfo().backgroundColor.w;
    break;
  }
  case '-': {
    if (gControlType == ctLightSource) {
      int light = gScene->getKernel()->getLight(gLampId);
      CPUPrimitive *p = gScene->getKernel()->getPrimitive(light);
      Material *m = gScene->getKernel()->getMaterial(p->materialId);
      if (m) {
        if (m->innerIllumination.x > 0.f) {
          m->innerIllumination.x -= 0.01f;
          gScene->getKernel()->compactBoxes(false);
        }
      }
    } else {
      gDistortion -= 0.01f;
      RayTracer::gKernel->setDistortion(gDistortion);
      LOG_INFO(1, "Distortion = " << gDistortion);
    }
    break;
  }
  case '\033':
  case '\015': {
    // Cleanup up and quit
    bNoPrompt = true;
    Cleanup(EXIT_SUCCESS);
    break;
  }
  }
  gScene->getSceneInfo().pathTracingIteration = 0;
}

/*
 ________________________________________________________________________________

 Mouse event handlers
 ________________________________________________________________________________
 */
void mouse(int button, int state, int x, int y) {
  SceneInfo &si = gScene->getSceneInfo();
  if (state == GLUT_DOWN) {
    mouse_buttons |= 1 << button;
    if (gDraft)
      // si.parameters.w = 1;
      si.renderBoxes = 2;
  } else {
    if (state == GLUT_UP) {
      mouse_buttons = 0;
      si.renderBoxes = 0;
      si.pathTracingIteration = 0;
      // si.parameters.w = 0;
    }
  }
  mouse_old_x = x;
  mouse_old_y = y;
}

/*
 ________________________________________________________________________________

 Mouse motion
 ________________________________________________________________________________
 */
void motion(int x, int y) {
  // gViewPos = gScene->getViewPos();
  // gViewDir = gScene->getViewDir();
  // gViewAngles = gScene->getViewAngles();

  gSelectedPrimitive = RayTracer::gKernel->getPrimitiveAt(x, y);

  SceneInfo &si = gScene->getSceneInfo();
  switch (mouse_buttons) {
  case 1: // Right Button
  {
    si.pathTracingIteration = 0;

    switch (gControlType) {
    case ctCamera: {
      // Camera Rotation
      float a = (mouse_old_x - x) / 100.f;
      if (fabs(a) <= 1.f)
        gViewAngles.y = gViewAngles.y - asin(a);
      a = (mouse_old_y - y) / 100.f;
      if (fabs(a) <= 1.f)
        gViewAngles.x = gViewAngles.x + asin(a);
      break;
    }
    case ctFocus: {
      // Depth of field focus
      gPostProcessingInfo.param2 += 50 * (mouse_old_y - y);
      LOG_INFO(1, "Param1=" << gPostProcessingInfo.param1);
      break;
    }
    case ctLightSource: {
      GPUKernel *kernel = gScene->getKernel();
      int light = kernel->getLight(gLampId);
      LOG_INFO(3, "Lamp " << gLampId << "[" << light << "] selected");
      if (light != -1) {
        CPUPrimitive *lamp = kernel->getPrimitive(light);
        lamp->p0.x -= 20 * (mouse_old_x - x);
        lamp->p0.z += 20 * (mouse_old_y - y);
        kernel->setPrimitiveCenter(light, lamp->p0);
        kernel->compactBoxes(false);
      } else
        LOG_ERROR("No light defined in the current scene");
      break;
    }
    case ctObject: {
      // Rotate primitives
      Vertex rotationCenter = {0.f, 0.f, 0.f};
      float a = (mouse_old_x - x) / 100.f;
      if (fabs(a) <= 1.f)
        gRotationAngles.y = asin(a);
      a = (mouse_old_y - y) / 100.f;
      if (fabs(a) <= 1.f)
        gRotationAngles.x = -asin(a);
      gRotationAngles.z += 0.f;
      gScene->rotatePrimitives(rotationCenter, gRotationAngles);
      RayTracer::gKernel->compactBoxes(false);
      break;
    }
    case ct3DVision: // Eye width
    {
      gScene->getSceneInfo().width3DVision += (mouse_old_y - y);
      break;
    }
    }
    break;
  }
  case 4: // Left Button
  {
    si.pathTracingIteration = 0;
    switch (gControlType) {
    case ctCamera: {
      // Translate camera distance
      gViewPos.z += 20.f * (mouse_old_y - y);
      gViewDir.z += 20.f * (mouse_old_y - y);
      break;
    }
    case ctLightSource: {
      // Translate Light source
      GPUKernel *kernel = gScene->getKernel();
      int light = kernel->getLight(gLampId);
      LOG_INFO(3, "Lamp " << gLampId << "[" << light << "] selected");
      if (light != -1) {
        CPUPrimitive *lamp = kernel->getPrimitive(light);
        lamp->p0.x -= 20 * (mouse_old_x - x);
        lamp->p0.y += 20 * (mouse_old_y - y);
        kernel->setPrimitiveCenter(light, lamp->p0);
        kernel->compactBoxes(false);
      } else
        LOG_ERROR("No light defined in the current scene");
      break;
    }
    case ctFocus: {
      // Depth of field focus
      gPostProcessingInfo.param1 += 20.f * (mouse_old_y - y);
      // gPostProcessingInfo.param3 = gPostProcessingInfo.param2/5;
      LOG_INFO(1, "Param2=" << gPostProcessingInfo.param2);
      break;
    }
    case ctFOV: {
      // Changing camera angle
      gViewPos.z -= 50 * (mouse_old_y - y);
      gViewAngles.w -= 50 * (mouse_old_x - x);
      LOG_INFO(1, "ViewPos  = " << gViewPos.x << "," << gViewPos.y << ","
                                << gViewPos.z);
      LOG_INFO(1, "ViewDir  = " << gViewDir.x << "," << gViewDir.y << ","
                                << gViewDir.z);
      LOG_INFO(1, "Distance = " << gScene->getSceneInfo().width3DVision);
      break;
    }
    case ctGroundHeight: {
      float h = gScene->getGroundHeight();
      h += 50 * (mouse_old_y - y);
      gScene->setGroundHeight(h);
      break;
    }
    }
    break;
  } break;
  }

  mouse_old_x = x;
  mouse_old_y = y;

  if (gAutoFocus) {
    // Autofocus
    GPUKernel *kernel = gScene->getKernel();
    int index = kernel->getPrimitiveAt(kernel->getSceneInfo().size.x / 2,
                                       kernel->getSceneInfo().size.y / 2);
    if (index != -1) {
      CPUPrimitive *p = kernel->getPrimitive(index);
      gViewDir.z = (p->p0.z + p->p1.z + p->p2.z) / 3.f;
    } else
      gViewDir.z = gViewPos.z + 8000.f;
  }
}

/*
 ________________________________________________________________________________

 Function to clean up and exit
 ________________________________________________________________________________
 */
void Cleanup(int iExitCode) {
#ifdef WIN32
  gTickCount = GetTickCount() - gTickCount;
  LOG_INFO(1, "----------------------------------------------------------------"
              "----------------");
  LOG_INFO(1, "Benchmark: " << gTickCount << " tick counts");
  LOG_INFO(1, "----------------------------------------------------------------"
              "----------------");
#endif // WIN32
  delete gScene->getKernel();
  delete gScene;
  exit(iExitCode);
}

#ifdef _USE_KINECT
/*
 ________________________________________________________________________________

 Create Kinect 3D Scene
 ________________________________________________________________________________
 */
void createKinectScene(int platform, int device) {
  std::cout << "createKinectScene" << std::endl;

  kernel = new kernel(false, gPlatform, gDevice);
  gScene->getSceneInfo().pathTracingIteration = 0;
  kernel.setSceneInfo(gScene->getSceneInfo());
  kernel.initBuffers();
  kernel.setCamera(gViewPos, gViewDir, gViewAngles);
  kernel.setPostProcessingInfo(gPostProcessingInfo);

  createRandomMaterials();

  kernel.render_begin(0.f);
  kernel.render_end((char *)gUbImage);

  char *depthBitmap = kernel.getDepthBitmap();
  if (depthBitmap != 0) {
    for (int x(0); x < KINECT_DEPTH_WIDTH; x += gKinectStep) {
      for (int y(0); y < KINECT_DEPTH_HEIGHT; y += gKinectStep) {
        int index = (y * KINECT_DEPTH_WIDTH + x) * KINECT_DEPTH_DEPTH;
        char a = depthBitmap[index];
        char b = depthBitmap[index + 1];
        int s = a * 256 + b;

        USHORT RealDepth = (s & 0xfff8) >> 3;
        // Masque pour retrouver si un player a �t� detect�
        USHORT Player = s & 7;
        // Transforme les informations de profondeur sur 13-Bit en une intensit�
        // cod� sur 8 Bits

        // afin d'afficher une couleur en fonction de la profondeur
        BYTE l = 255 - (BYTE)(256 * RealDepth / 0x0fff);

        int p = kernel.addPrimitive(ptSphere);
        kernel.setPrimitive(p, 5 + p / gKinectNbSpherePerBox, (x - 160) * 100.f,
                            (y - 120) * 100.f, 0.f, gKinectStep * 10.f, 0.f,
                            0.f, 20 + x % 20, 1, 1);
      }
    }
  }

  // Sol
  int s0 = kernel.addPrimitive(ptXZPlane);
  kernel.setPrimitive(s0, 0, 0.f, -5000.f, 0.f, 10000.f, 0.f, 10000.f, 100, 1,
                      1);

  // Wall
  int m_nbPrimitives = kernel.addPrimitive(ptXYPlane);
  kernel.setPrimitive(m_nbPrimitives, 0, 0.f, 0.f, 5000.f, 5000.f, 5000.f, 0.f,
                      110, 1, 1);
  m_nbPrimitives = kernel.addPrimitive(ptYZPlane);
  kernel.setPrimitive(m_nbPrimitives, 1, -5000.f, 0.f, 0.f, 0.f, 5000.f, 5000.f,
                      107, 1, 1);
  m_nbPrimitives = kernel.addPrimitive(ptYZPlane);
  kernel.setPrimitive(m_nbPrimitives, 2, 5000.f, 0.f, 0.f, 0.f, 5000.f, 5000.f,
                      108, 1, 1);
  m_nbPrimitives = kernel.addPrimitive(ptXZPlane);
  kernel.setPrimitive(m_nbPrimitives, 3, 0.f, 5000.f, 0.f, 5000.f, 0.f, 5000.f,
                      110, 1, 1);

  // Camera
  m_nbPrimitives = kernel.addPrimitive(ptCamera);
  kernel.setPrimitive(m_nbPrimitives, 0, -2500.f, -4000.f, -5000.f, 1500.f,
                      1500.f, 0.f, DEFAULT_LIGHT_MATERIAL, 1, 1);

  // Lights
  gLampIndex = kernel.addPrimitive(ptSphere);
  kernel.setPrimitive(gLampIndex, 4, -5000.f, 2000.f, -20000.f, 500, 0, 50, 128,
                      1, 1);

  gNbBoxes = kernel.compactBoxes();
}
#endif // USE_KINECT
       /*
        ________________________________________________________________________________

        Create 3D Scene
        ________________________________________________________________________________
        */
void createScene() {
  gAnimate = false;

  LOG_INFO(1, "Creating Scene (" << gTest << ") ...");
  gHint = "Camera";
  switch (gTest) {
#ifdef _USE_KINECT
  case 0:
    gScene =
        new KinectFaceTrackingScene("Kinect Face Tracking", gOptimalNbOfBoxes);
    break;
#else
  case 0:
    // gScene = new CylinderScene("Cylinders", gOptimalNbOfBoxes);
    gScene = new SwcScene("Neuron Morphology", gOptimalNbOfBoxes);
    // gScene = new ColumnScene("Neuron Column", gOptimalNbOfBoxes);
    break;
#endif
  case 1:
    gScene = new TrianglesScene("IRT models", gOptimalNbOfBoxes);
    break;
  case 2:
    gScene = new ObjScene("OBJ models", gOptimalNbOfBoxes, gFilename);
    break;
  case 3:
    gScene = new MoleculeScene("Molecules", gOptimalNbOfBoxes);
    break;
  case 4:
    gScene = new FractalScene("Fractals", gOptimalNbOfBoxes);
    break;
  case 5:
    gScene = new PerpetualMotionScene("Perpetual Motion", gOptimalNbOfBoxes);
    break;
  case 6:
    gScene = new XmasScene("Christmas", gOptimalNbOfBoxes);
    break;
  case 7:
    gScene = new Year2013("Happy new year 2013", gOptimalNbOfBoxes);
    break;
  case 8:
    gScene = new DoggyStyleScene("Doggy style", gOptimalNbOfBoxes);
    break;
  case 9:
    gScene = new MetaballsScene("Metaballs", gOptimalNbOfBoxes);
    gAnimate = true;
    break;
  case 10:
    gScene = new GraphScene("Charts", gOptimalNbOfBoxes);
    break;
  case 11:
    gScene = new CornellBoxScene("Cornell Box", gOptimalNbOfBoxes);
    break;
  case 12:
    gScene = new WaterScene("Water", gOptimalNbOfBoxes);
    break;
  case 13:
    gScene = new CubesScene("Cubes", gOptimalNbOfBoxes);
    break;
  case 14:
    gScene = new TransparentScene("Transparent", gOptimalNbOfBoxes);
    break;
  }
  LOG_INFO(1, "Scene created");
  RayTracer::gKernel->setOptimalNbOfBoxes(gOptimalNbOfBoxes);
  RayTracer::gKernel->setTreeDepth(gTreeDepth);
  gScene->setCornellBox(gCornellBoxType);
  gScene->setCurrentModel(m_counter);
  LOG_INFO(1, "Initializing Raytracer...");
  gScene->initialize(gWindowWidth, gWindowHeight);
  gScene->getKernel()->setCamera(gViewPos, gViewDir, gViewAngles);
}

/*
 ________________________________________________________________________________

 Main
 ________________________________________________________________________________
 */
#ifndef WIN32
int main(int argc, char *argv[]) {
  LOG_INFO(1, "Starting program...");
  std::vector<std::string> arguments;
  for (int i(0); i < argc; ++i) {
    arguments.push_back(argv[i]);
  }
  readParameters(arguments);

  srand(static_cast<int>(time(0)));

  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA
  // interop.
  LOG_INFO(1, "Initializing OpenGL...");
  initgl(argc, argv);

  // initMenus();

  atexit(cleanup);

  // Create Scene
  createScene();

  // glutFullScreen();
  LOG_INFO(1, "Entering OpenGL loop...");
  glutMainLoop();

  // Normally unused return path
  Cleanup(EXIT_SUCCESS);

  return 0;
}
#else
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                     LPSTR lpCmdLine, int nCmdShow) {
  LOG_INFO(1, "Command line: " << lpCmdLine);
  LPWSTR *szArglist;
  int nArgs;
  szArglist = CommandLineToArgvW(GetCommandLineW(), &nArgs);

  srand(static_cast<int>(time(0)));
  std::vector<std::string> arguments;
  for (int i(0); i < nArgs; ++i) {
    std::wstring a(szArglist[i]);
    std::string argument(a.begin(), a.end());
    arguments.push_back(argument);
  }

  readParameters(arguments);

  // First initialize OpenGL context, so we can properly set the GL for CUDA.
  // This is necessary in order to achieve optimal performance with OpenGL/CUDA
  // interop.
  initgl(nArgs, (char **)szArglist);

  // initMenus();

  atexit(cleanup);

  // Create Scene
  createScene();

  // Benchmark
  gTickCount = GetTickCount();

  // glutFullScreen();
  glutMainLoop();

  // Normally unused return path
  Cleanup(EXIT_SUCCESS);

  return 0;
}
#endif // WIN32
