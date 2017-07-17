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

#ifdef WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#endif // WIN32

#include <algorithm>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string>

#include <Consts.h>
#include <Logging.h>

#include <io/FileMarshaller.h>
#include <io/OBJReader.h>
#include <io/PDBReader.h>

#include <iostream>

#include "ObjScene.h"

int m_counter = 0;

ObjScene::ObjScene(const std::string& name, const int nbMaxPrimitivePerBox, const std::string& filename)
    : m_filename(filename)
    , Scene(name, nbMaxPrimitivePerBox)
{
    m_currentModel = 0;
    m_groundHeight = -2500.f;
}

ObjScene::~ObjScene(void)
{
}

void ObjScene::doInitialize()
{
    const float s = 10000.f;
    Vertex objectPosition = {0.f, 0.f, 0.f};
    m_objectScale.x = s;
    m_objectScale.y = s;
    m_objectScale.z = s;

    // Scene Bounding Box
    SceneInfo& sceneInfo = m_gpuKernel->getSceneInfo();
    CPUBoundingBox AABB;
    AABB.parameters[0].x = sceneInfo.viewDistance;
    AABB.parameters[0].y = sceneInfo.viewDistance;
    AABB.parameters[0].z = sceneInfo.viewDistance;
    AABB.parameters[1].x = -sceneInfo.viewDistance;
    AABB.parameters[1].y = -sceneInfo.viewDistance;
    AABB.parameters[1].z = -sceneInfo.viewDistance;

    if (m_filename.length() != 0)
    {
        LOG_INFO(1, "Loading " << m_filename);
        OBJReader objectReader;
        CPUBoundingBox aabb;
        CPUBoundingBox inAABB;
        memset(&inAABB, 0, sizeof(CPUBoundingBox));
        Vertex size = objectReader.loadModelFromFile(m_filename, *m_gpuKernel, objectPosition, true, m_objectScale,
                                                     true, 1000, false, true, aabb, false, inAABB);
        m_groundHeight = -size.y / 2.f * m_objectScale.y - EPSILON;
    }
    else
    {
        std::vector<std::string> fileNames;
#ifdef WIN32
        // filename vector
        HANDLE hFind(nullptr);
        WIN32_FIND_DATA FindData;

        std::string fullFilter("../medias/obj/*.obj");

        hFind = FindFirstFile(fullFilter.c_str(), &FindData);
        if (hFind != INVALID_HANDLE_VALUE)
        {
            do
            {
                if (strlen(FindData.cFileName) != 0)
                    fileNames.push_back(FindData.cFileName);
            } while (FindNextFile(hFind, &FindData));
        }
#else
        std::string path = "../medias/obj";
        DIR* dp;
        struct dirent* dirp;
        if ((dp = opendir(path.c_str())) == NULL)
        {
            LOG_ERROR(errno << " opening " << path);
        }
        else
        {
            while ((dirp = readdir(dp)) != NULL)
            {
                std::string filename(dirp->d_name);
                if (filename != "." && filename != ".." && filename.find(".obj") != std::string::npos &&
                    filename.find(".mtl") == std::string::npos)
                {
                    std::string fullPath(path);
                    fullPath += "/";
                    fullPath += dirp->d_name;
                    LOG_INFO(1, "Model: " << fullPath);
                    fileNames.push_back(fullPath);
                }
            }
            closedir(dp);
        }
#endif // WIN32
    }
}

void ObjScene::doPostInitialize()
{
}

void ObjScene::doAnimate()
{
}

void ObjScene::doAddLights()
{
    // lights
    if (m_gpuKernel->getNbActiveLamps() == 0)
    {
        m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
        m_gpuKernel->setPrimitive(m_nbPrimitives, 1000.f, 8000.f, -1000.f, 10.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL);
    }
}
