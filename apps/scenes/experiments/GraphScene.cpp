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

#include "GraphScene.h"

#include <opengl/rtgl.h>

#ifdef WIN32
#include <windows.h>
#else
#include <fstream>
#include <math.h>
#include <stdlib.h>
#endif

using namespace solr;

GraphScene::GraphScene(const std::string& name, const int nbMaxPrimitivePerBox)
    : Scene(name, nbMaxPrimitivePerBox)
{
}

GraphScene::~GraphScene(void)
{
    delete m_graphValues;
}

void GraphScene::buildGraph(bool update)
{
    if (!update)
        // Graph
        for (int i(0); i < m_nbGraphElements; ++i)
        {
            m_nbPrimitives = m_gpuKernel->addPrimitive(ptCylinder);
            m_gpuKernel->setPrimitive(m_nbPrimitives, -1000.f, -5000.f, 0.f, -1000.f, 1000.f, 0.f, 20.f, 0.f, 0.f,
                                      m_graphMaterial);

            if (i == 0)
                m_startGraph = m_nbPrimitives;

            m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
            m_gpuKernel->setPrimitive(m_nbPrimitives, -1000.f, 1000.f, 0.f, 20.f, 0.f, 0.f, m_graphMaterial);
        }

    float x = m_graphSpace * m_nbGraphElements;
    float y = -5000.f;
    for (int i(0); i < m_nbGraphElements; ++i)
    {
        m_graphValues[i] = (cos(m_gpuKernel->getSceneInfo().misc.y * 0.00402f + i * 0.2f) *
                            sin(m_gpuKernel->getSceneInfo().misc.y * 0.00208f + i * 0.5f)) *
                           2000.f;

        int material = m_graphMaterial; //+i;
        switch (m_graphMode)
        {
        case 0:
        {
            float z = 4.f * m_graphSize.x * (i / (m_nbGraphElements / 2)) - 2.f * m_graphSize.x;
            int j = (i * 2) % (m_nbGraphElements / 2);
            m_gpuKernel->setPrimitive(m_startGraph + (i * 2), m_graphSpace + j * 4.f * m_graphSpace - x, y, z,
                                      m_graphSpace + j * 4.f * m_graphSpace - x, m_graphValues[i], z, m_graphSize.x,
                                      0.f, 0.f, material);

            m_gpuKernel->setPrimitive(m_startGraph + (i * 2) + 1, m_graphSpace + j * 4.f * m_graphSpace - x,
                                      m_graphValues[i], z, m_graphSize.x, 0.f, 0.f, material);

            break;
        }
        case 1:
        {
            if (i > 0)
            {
                m_gpuKernel->setPrimitive(m_startGraph + (i * 2), m_graphSpace / 2.f + (i - 1) * 2.f * m_graphSpace - x,
                                          m_graphValues[i - 1], 0.f, m_graphSpace / 2.f + i * 2.f * m_graphSpace - x,
                                          m_graphValues[i], 0.f, m_graphSize.x / 2.f, 0.f, 0.f, material);
            }
            else
            {
                m_gpuKernel->setPrimitive(m_startGraph + (i * 2), 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                          material);
            }
            m_gpuKernel->setPrimitive(m_startGraph + (i * 2) + 1, m_graphSpace / 2.f + i * 2.f * m_graphSpace - x,
                                      m_graphValues[i], 0.f, m_graphSize.x / 2.f, 0.f, 0.f, material);
            break;
        }
        case 2:
        {
            if (i > 0)
            {
                m_gpuKernel->setPrimitive(m_startGraph + (i * 2), m_graphSpace / 2.f + (i - 1) * 2.f * m_graphSpace - x,
                                          m_graphValues[i - 1], 0.f, m_graphSpace / 2.f + i * 2.f * m_graphSpace - x,
                                          m_graphValues[i], 0.f, m_graphSize.x / 4.f, 0.f, 0.f, material);
            }
            else
            {
                m_gpuKernel->setPrimitive(m_startGraph + (i * 2), 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                          material);
            }
            m_gpuKernel->setPrimitive(m_startGraph + (i * 2) + 1, m_graphSpace / 2.f + i * 2.f * m_graphSpace - x,
                                      m_graphValues[i], 0.f, m_graphSize.x, 0.f, 0.f, material);
            break;
        }
        }
    }
    m_gpuKernel->rotatePrimitives(m_rotationCenter, m_rotationAngles);
}

void GraphScene::buildChart(const std::string& filename)
{
    m_valueSize.y = 0;
    m_valueSize.x = 0;
    float maxY(0.f);
    std::ifstream file(filename.c_str());
    if (file.is_open())
    {
        while (file.good())
        {
            std::string str;
            std::getline(file, str);
            size_t p;
            int line(0);
            do
            {
                p = str.find(";");
                std::string value = str.substr(0, p);
                float y(static_cast<float>(atof(value.c_str())));
                m_values[line][m_valueSize.x] = y;
                if (y > maxY)
                    maxY = y;
                str = str.substr(p + 1, str.length());
                ++line;
                if (line > m_valueSize.y)
                    m_valueSize.y = line;
            } while (p != std::string::npos);
            // Last value of the line
            float y(static_cast<float>(atof(str.c_str())));
            m_values[line][m_valueSize.x] = y;

            ++m_valueSize.x;
        }
        file.close();
    }
    else
    {
        LOG_ERROR("Failed to open " << filename.c_str());
    }

    LOG_INFO(1, "Chart size: " << m_valueSize.x << "x" << m_valueSize.y);
    m_graphSize.x = 4000.f;
    m_graphSize.y = 4000.f;
    m_graphSize.z = 4000.f;

    m_graphScale.x = 2.f * m_graphSize.x / m_valueSize.y;
    m_graphScale.y = m_graphSize.y / maxY;
    m_graphScale.z = 2.f * m_graphSize.z / m_valueSize.x;

    m_graphCenter.x = m_valueSize.y / 2.f;
    m_graphCenter.y = -maxY / 2.f;
    m_graphCenter.z = m_valueSize.x / 2.f;

    glBegin(GL_TRIANGLES);
    m_gpuKernel->setCurrentMaterial(40);
    // Right border
    glVertex3f(m_graphSize.x, -m_graphSize.y, -m_graphSize.z);
    glVertex3f(m_graphSize.x, -m_graphSize.y, m_graphSize.z);
    glVertex3f(m_graphSize.x, m_graphSize.y, m_graphSize.z);

    glVertex3f(m_graphSize.x, m_graphSize.y, m_graphSize.z);
    glVertex3f(m_graphSize.x, m_graphSize.y, -m_graphSize.z);
    glVertex3f(m_graphSize.x, -m_graphSize.y, -m_graphSize.z);

    // Back
    glVertex3f(-m_graphSize.x, -m_graphSize.y, m_graphSize.z);
    glVertex3f(m_graphSize.x, -m_graphSize.y, m_graphSize.z);
    glVertex3f(m_graphSize.x, m_graphSize.y, m_graphSize.z);

    glVertex3f(m_graphSize.x, m_graphSize.y, m_graphSize.z);
    glVertex3f(-m_graphSize.x, m_graphSize.y, m_graphSize.z);
    glVertex3f(-m_graphSize.x, -m_graphSize.y, m_graphSize.z);

    // Ground
    glVertex3f(-m_graphSize.x, -m_graphSize.y, -m_graphSize.z);
    glVertex3f(m_graphSize.x, -m_graphSize.y, -m_graphSize.z);
    glVertex3f(m_graphSize.x, -m_graphSize.y, m_graphSize.z);

    glVertex3f(m_graphSize.x, -m_graphSize.y, m_graphSize.z);
    glVertex3f(-m_graphSize.x, -m_graphSize.y, m_graphSize.z);
    glVertex3f(-m_graphSize.x, -m_graphSize.y, -m_graphSize.z);
    glEnd();

    int m(0);
    for (int z(0); z < m_valueSize.x; ++z)
    {
        glBegin(GL_TRIANGLES);
        glTranslatef(-m_graphScale.x * m_graphCenter.x, m_graphScale.y * m_graphCenter.y,
                     -m_graphScale.z * m_graphCenter.z);
        m_gpuKernel->setCurrentMaterial(41 + m);
        for (int x(0); x < m_valueSize.y; ++x)
        {
            LOG_INFO(1, x << "," << m_values[x][z] << "," << z);

            glVertex3f(m_graphScale.x * x, m_graphScale.y * m_values[x][z], -m_graphScale.z * z);
            glVertex3f(m_graphScale.x * (x + 1), m_graphScale.y * m_values[x + 1][z], -m_graphScale.z * z);
            glVertex3f(m_graphScale.x * (x + 1), m_graphScale.y * m_values[x + 1][z + 1], -m_graphScale.z * (z + 1));

            glVertex3f(m_graphScale.x * (x + 1), m_graphScale.y * m_values[x + 1][z + 1], -m_graphScale.z * (z + 1));
            glVertex3f(m_graphScale.x * x, m_graphScale.y * m_values[x][z + 1], -m_graphScale.z * (z + 1));
            glVertex3f(m_graphScale.x * x, m_graphScale.y * m_values[x][z], -m_graphScale.z * z);
        }
        glEnd();
        ++m;

        if (false)
        {
            // Labels
            for (int x(0); x < m_valueSize.y; ++x)
            {
                int p = m_gpuKernel->addPrimitive(ptXYPlane);
                m_gpuKernel->setPrimitive(p, -m_graphScale.x * m_graphCenter.x + m_graphScale.x * x,
                                          500.f + m_graphScale.y * m_graphCenter.y + m_graphScale.y * m_values[x][z],
                                          m_graphScale.z * m_graphCenter.z - m_graphScale.z * z, 200.f, 100.f, 0.f,
                                          119);
            }
        }
    }

    float lineSize(20.f);
    for (int z(-m_valueSize.x / 2); z <= m_valueSize.x / 2; ++z)
    {
        glBegin(GL_LINES);
        m_gpuKernel->setCurrentMaterial(42);
        glPointSize(lineSize);
        glVertex3f(-m_graphSize.x, -m_graphSize.y, m_graphScale.z * z);
        glVertex3f(m_graphSize.x, -m_graphSize.y, m_graphScale.z * z);
        glEnd();

        glBegin(GL_LINES);
        m_gpuKernel->setCurrentMaterial(42);
        glPointSize(lineSize);
        glVertex3f(m_graphSize.x, -m_graphSize.y, m_graphScale.z * z);
        glVertex3f(m_graphSize.x, m_graphSize.y, m_graphScale.z * z);
        glEnd();
    }

    // for( int y(-maxY/2); y<=maxY/2; ++y )
    for (int y(-5); y <= 5; ++y)
    {
        glBegin(GL_LINES);
        glPointSize(lineSize);
        glVertex3f(-m_graphSize.x, m_graphSize.y * y / 5.f, m_graphSize.z);
        glVertex3f(m_graphSize.x, m_graphSize.y * y / 5.f, m_graphSize.z);
        glEnd();

        glBegin(GL_LINES);
        glPointSize(lineSize);
        glVertex3f(m_graphSize.x, m_graphSize.y * y / 5.f, -m_graphSize.z);
        glVertex3f(m_graphSize.x, m_graphSize.y * y / 5.f, m_graphSize.z);
        glEnd();
    }

    for (int x(-m_valueSize.y / 2); x <= m_valueSize.y / 2; ++x)
    {
        m_gpuKernel->setCurrentMaterial(42);
        glBegin(GL_LINES);
        glPointSize(lineSize);
        glVertex3f(m_graphScale.x * x, -m_graphSize.y, m_graphSize.z);
        glVertex3f(m_graphScale.x * x, -m_graphSize.y, -m_graphSize.z);
        glEnd();

        glBegin(GL_LINES);
        glPointSize(lineSize);
        glVertex3f(m_graphScale.x * x, -m_graphSize.y, m_graphSize.z);
        glVertex3f(m_graphScale.x * x, m_graphSize.y, m_graphSize.z);
        glEnd();
    }
}

void GraphScene::doInitialize()
{
    // Attributes
    m_nbGraphElements = 10;
    m_graphValues = new float[m_nbGraphElements];
    m_startGraph = 0;
    m_graphSpace = 500.f;
    m_graphSize.x = 300.f;
    m_graphMaterial = 13;
    m_graphMode = rand() % 3;
    m_graphObjectsPerBox = 5;

    m_groundHeight = -5000.f;

    buildChart(std::string(DEFAULT_MEDIA_FOLDER) + "/charts/data.csv");
}

void GraphScene::doAnimate()
{
}

void GraphScene::doAddLights()
{
    // Lights
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
    m_gpuKernel->setPrimitive(m_nbPrimitives, 0.f, 5000.f, 0.f, 50.f, 0, 0, DEFAULT_LIGHT_MATERIAL);
}
