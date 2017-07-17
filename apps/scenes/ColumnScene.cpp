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

#include <math.h>
#include <stdlib.h>

#include <FileMarshaller.h>
#include <Logging.h>
#include <OBJReader.h>

#include "ColumnScene.h"

ColumnScene::ColumnScene(const std::string& name, const int nbMaxPrimitivePerBox)
    : Scene(name, nbMaxPrimitivePerBox)
{
}

ColumnScene::~ColumnScene(void)
{
}

/*
________________________________________________________________________________

Create simple scene with cylinders
________________________________________________________________________________
*/
void ColumnScene::doInitialize()
{
    const int percentage = 1;
    const float s = 10.f;
    const float boxScale = 100.f;

    std::vector<std::string> fileNames;
    std::string path("E:\\HBP\\Meshes");

    Vertex objectPosition = {0.f, 0.f, 0.f};
    m_objectScale.x = s;
    m_objectScale.y = s;
    m_objectScale.z = s;

    // Scene Bounding Box
    SceneInfo& sceneInfo = m_gpuKernel->getSceneInfo();
    CPUBoundingBox AABB;
    AABB.parameters[0].x = sceneInfo.viewDistance.x;
    AABB.parameters[0].y = sceneInfo.viewDistance.x;
    AABB.parameters[0].z = sceneInfo.viewDistance.x;
    AABB.parameters[1].x = -sceneInfo.viewDistance.x;
    AABB.parameters[1].y = -sceneInfo.viewDistance.x;
    AABB.parameters[1].z = -sceneInfo.viewDistance.x;

#ifdef WIN32
    // filename vector
    HANDLE hFind(nullptr);
    WIN32_FIND_DATA FindData;

    std::string fullFilter(path + "\\*.obj");

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

    const std::string star("neuron_12558.obj");
    OBJReader objectReader;
    CPUBoundingBox inAABB;
    memset(&inAABB, 0, sizeof(CPUBoundingBox));
    Vertex size = objectReader.loadModelFromFile(path + "\\" + star, *m_gpuKernel, objectPosition, false, m_objectScale,
                                                 false, 10, false, false, AABB, false, inAABB);

    Vertex halfSize = {
        (AABB.parameters[1].x - AABB.parameters[0].x) / 2.f, (AABB.parameters[1].y - AABB.parameters[0].y) / 2.f,
        (AABB.parameters[1].z - AABB.parameters[0].z) / 2.f,
    };

    Vertex boxCenter = {(AABB.parameters[0].x + AABB.parameters[1].x) / 2.f,
                        (AABB.parameters[0].y + AABB.parameters[1].y) / 2.f,
                        (AABB.parameters[0].z + AABB.parameters[1].z) / 2.f};

    inAABB.parameters[0].x = (boxCenter.x - halfSize.x * boxScale);
    inAABB.parameters[0].y = (boxCenter.y - halfSize.y * boxScale);
    inAABB.parameters[0].z = (boxCenter.z - halfSize.y * boxScale);
    inAABB.parameters[1].x = (boxCenter.x + halfSize.x * boxScale);
    inAABB.parameters[1].y = (boxCenter.y + halfSize.y * boxScale);
    inAABB.parameters[1].z = (boxCenter.z + halfSize.z * boxScale);

    m_actorPosition = boxCenter;
    LOG_INFO(1, "Actor Position: " << m_actorPosition.x << "," << m_actorPosition.y << "," << m_actorPosition.z << ")");

    for (int i(0); i < fileNames.size(); ++i)
    {
        // if( rand()%100<percentage )
        {
            m_currentModel = i;
#ifdef WIN32
            m_name = path + "\\" + fileNames[m_currentModel];
#else
            m_name = fileNames[m_currentModel];
#endif // WIN32
            LOG_INFO(1, "--- Loading " << m_name << " ---");
            if (m_name.find(star) == -1)
            {
                CPUBoundingBox aabb;
                objectReader.loadModelFromFile(m_name, *m_gpuKernel, objectPosition, false, m_objectScale, false, 11,
                                               false, false, aabb, true, inAABB);

                m_groundHeight = -size.y / 2.f - EPSILON;

                if (aabb.parameters[0].x < AABB.parameters[0].x)
                    AABB.parameters[0].x = aabb.parameters[0].x;
                if (aabb.parameters[0].y < AABB.parameters[0].y)
                    AABB.parameters[0].y = aabb.parameters[0].y;
                if (aabb.parameters[0].z < AABB.parameters[0].z)
                    AABB.parameters[0].z = aabb.parameters[0].z;

                if (aabb.parameters[1].x > AABB.parameters[1].x)
                    AABB.parameters[1].x = aabb.parameters[1].x;
                if (aabb.parameters[1].y > AABB.parameters[1].y)
                    AABB.parameters[1].y = aabb.parameters[1].y;
                if (aabb.parameters[1].z > AABB.parameters[1].z)
                    AABB.parameters[1].z = aabb.parameters[1].z;
            }
        }
    }

    // Save into binary format
    FileMarshaller fm;
    fm.saveToFile(*m_gpuKernel, "column.irt");
}

void ColumnScene::doAnimate()
{
    m_rotationAngles.x = 0.02f;
    m_rotationAngles.y = 0.01f;
    m_rotationAngles.z = 0.015f;
    m_gpuKernel->rotatePrimitives(m_rotationCenter, m_rotationAngles);
    m_gpuKernel->compactBoxes(false);
}

void ColumnScene::doAddLights()
{
    for (int i(0); i < 5; ++i)
    {
        // lights
        m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
        m_gpuKernel->setPrimitive(m_nbPrimitives, rand() % 20000 - 10000.f, rand() % 5000 - m_groundHeight,
                                  rand() % 20000 - 10000.f, 10.f, 0.f, 0.f, 120 + i);
        m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives, false);
    }
}
