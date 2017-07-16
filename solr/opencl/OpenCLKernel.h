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

#pragma once

#include "types.h"

#include <stdio.h>
#include <string>

#ifdef WIN32
#include <windows.h>
#endif // WIN32

#if USE_KINECT
#include <nuiapi.h>
#endif // USE_KINECT

typedef cl_float4 Vertex;
typedef cl_int4 PrimitiveXYIdBuffer;
typedef unsigned char BitmapBuffer;
typedef cl_float RandomBuffer;
typedef cl_int Lamp;

#include "../Consts.h"
#include "../GPUKernel.h"

const long MAX_DEVICES = 10;

class SOLR_API OpenCLKernel : public GPUKernel {

public:
  enum KernelSourceType { kst_file, kst_string };

public:
  OpenCLKernel(const bool activeLogging, const int optimalNbOfPrimmitivesPerBox,
               const int platform, const int device);
  ~OpenCLKernel();

  virtual void initBuffers();

public:
  // ---------- Devices ----------
  void initializeDevice();
  void releaseDevice();

  void recompileKernels(const std::string &kernelCode = "");
  void releaseKernels();

  virtual void reshape();

public:
  // ---------- Rendering ----------
  void render_begin(const float timer);
  void render_end();

public:
  virtual std::string getGPUDescription();

public:
  static void populateOpenCLInformation();
  static int getNumPlatforms();
  static std::string getPlatformDescription(const int platform);
  static int getNumDevices(const int platform);
  static std::string getDeviceDescription(const int platform, const int device);

private:
  // Platforms
  static cl_platform_id m_platforms[MAX_DEVICES];
  static cl_uint m_numberOfPlatforms;
  static std::string m_platformsDescription[MAX_DEVICES];

  // Devices
  static cl_device_id m_devices[MAX_DEVICES][MAX_DEVICES];
  static cl_uint m_numberOfDevices[MAX_DEVICES];
  static std::string m_devicesDescription[MAX_DEVICES][MAX_DEVICES];

private:
  // OpenCL Objects
  int m_platform;
  int m_device;
  cl_device_id m_hDeviceId;
  cl_context m_hContext;
  cl_command_queue m_hQueue;
  cl_uint m_preferredWorkGroupSize;

private:
  // Rendering kernels
  cl_kernel m_kStandardRenderer;
  cl_kernel m_kAnaglyphRenderer;
  cl_kernel m_k3DVisionRenderer;
  cl_kernel m_kFishEyeRenderer;
  cl_kernel m_kVolumeRenderer;

  // Post processing kernels
  cl_kernel m_kDefault;
  cl_kernel m_kDepthOfField;
  cl_kernel m_kAmbientOcclusion;
  cl_kernel m_kRadiosity;
  cl_kernel m_kFilter;

private:
  cl_mem m_dBoundingBoxes;
  cl_mem m_dPrimitives;
  cl_mem m_dLamps;
  cl_mem m_dLightInformation;
  cl_mem m_dMaterials;
  cl_mem m_dTextures;
  cl_mem m_dRandoms;
  cl_mem m_dBitmap;
  cl_mem m_dPostProcessingBuffer;
  cl_mem m_dPrimitivesXYIds;

private:
  int *m_hPrimitiveIDs;

#ifdef KINECT
private:
  cl_mem m_dVideo;
  cl_mem m_dDepth;
#endif // KINECT
};
