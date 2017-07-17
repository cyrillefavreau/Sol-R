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

#include <iostream>
#include <math.h>

#include <opengl/rtgl.h>
using namespace RayTracer;

#include "MetaballsScene.h"

#ifdef USE_LEAPMOTION
#pragma comment(lib, "Leap.lib")
#endif // USE_LEAPMOTION

const int minGridSize = 40;
float threshold = 1.0f;

#ifdef USE_KINECT
int gridSize = 60;
const int numMetaballs = 20;
Vertex size = {numMetaballs * 10.f, numMetaballs * 10.f, numMetaballs * 10.f};
Vertex amplitude = {size.x / 5.f, size.y / 5.f, size.z / 5.f};
Vertex scale = {2500.f / numMetaballs, 2500.f / numMetaballs, 2500.f / numMetaballs};
#else
int gridSize = 60;
const int numMetaballs = 10;
Vertex size = {numMetaballs * 5.f, numMetaballs * 5.f, numMetaballs * 5.f};
Vertex amplitude = {size.x / 5.f, size.y / 5.f, size.z / 5.f};
Vertex scale = {2000.f / numMetaballs, 2000.f / numMetaballs, 2000.f / numMetaballs};
#endif // USE_KINECT
METABALL metaballs[numMetaballs];

MetaballsScene::MetaballsScene(const std::string& name, const int nbMaxPrimitivePerBox)
    : Scene(name, nbMaxPrimitivePerBox)
    , m_timer(0)
    , numCubes(0)
    , numFacesDrawn(0)
    , numVertices(0)
#ifdef USE_LEAPMOTION
    , m_leapMotionController(nullptr)
#endif // USE_LEAPMOTION
{
    // initialise the metaBall_ size and positions
    vertices = new CUBE_GRID_VERTEX[(maxGridSize + 1) * (maxGridSize + 1) * (maxGridSize + 1)];
    cubes = new CUBE_GRID_CUBE[maxGridSize * maxGridSize * maxGridSize];
}

MetaballsScene::~MetaballsScene(void)
{
#ifdef USE_LEAPMOTION
    finalizeLeapMotion();
#endif // USE_LEAPMOTION

    if (vertices)
        delete[] vertices;
    vertices = NULL;
    numVertices = 0;

    if (cubes)
        delete[] cubes;
    cubes = NULL;
    numCubes = 0;
}

void MetaballsScene::Init()
{
    /*
   m_viewPos.y = 1000.f;
   m_viewPos.z = -15000.f;
   m_viewDir.y = m_viewPos.y;
   m_viewDir.z = m_viewPos.z+5000.f;
   */

    // VERTICES
    numVertices = (gridSize + 1) * (gridSize + 1) * (gridSize + 1);

    int currentVertex = 0;

    for (int i = 0; i < gridSize + 1; i++)
    {
        for (int j = 0; j < gridSize + 1; j++)
        {
            for (int k = 0; k < gridSize + 1; k++)
            {
                vertices[currentVertex].position.x = (i * size.x) / (gridSize)-size.x / 2.f;
                vertices[currentVertex].position.y = (j * size.y) / (gridSize)-size.y / 2.f;
                vertices[currentVertex].position.z = (k * size.z) / (gridSize)-size.z / 2.f;

                vertices[currentVertex].textCoords.x = static_cast<float>(i) / static_cast<float>(gridSize);
                vertices[currentVertex].textCoords.y = static_cast<float>(j) / static_cast<float>(gridSize);
                ;
                vertices[currentVertex].textCoords.z = 0.f;

                currentVertex++;
            }
        }
    }

    // CUBES
    numCubes = (gridSize) * (gridSize) * (gridSize);

    int currentCube = 0;

    for (int i = 0; i < gridSize; i++)
    {
        for (int j = 0; j < gridSize; j++)
        {
            for (int k = 0; k < gridSize; k++)
            {
                cubes[currentCube].vertices[0] = &vertices[(i * (gridSize + 1) + j) * (gridSize + 1) + k];
                cubes[currentCube].vertices[1] = &vertices[(i * (gridSize + 1) + j) * (gridSize + 1) + k + 1];
                cubes[currentCube].vertices[2] = &vertices[(i * (gridSize + 1) + (j + 1)) * (gridSize + 1) + k + 1];
                cubes[currentCube].vertices[3] = &vertices[(i * (gridSize + 1) + (j + 1)) * (gridSize + 1) + k];
                cubes[currentCube].vertices[4] = &vertices[((i + 1) * (gridSize + 1) + j) * (gridSize + 1) + k];
                cubes[currentCube].vertices[5] = &vertices[((i + 1) * (gridSize + 1) + j) * (gridSize + 1) + k + 1];
                cubes[currentCube].vertices[6] =
                    &vertices[((i + 1) * (gridSize + 1) + (j + 1)) * (gridSize + 1) + k + 1];
                cubes[currentCube].vertices[7] = &vertices[((i + 1) * (gridSize + 1) + (j + 1)) * (gridSize + 1) + k];

                currentCube++;
            }
        }
    }

    m_groundHeight = -4000.f;
}

void MetaballsScene::doInitialize()
{
#ifdef USE_LEAPMOTION
    initializeLeapMotion();
#endif // USE_LEAPMOTION

    m_timer = 0;
    // set up metaballs
    for (int i = 0; i < numMetaballs; i++)
    {
        Vertex zero = {0.f, 0.f, 0.f};
        metaballs[i].Init(zero, 5.0f + float(i));
    }
}

void MetaballsScene::doAnimate()
{
    m_gpuKernel->resetFrame();

    // update balls' position
    for (int i(0); i < numMetaballs; ++i)
    {
        // float timer = static_cast<float>(i*10+m_timer*m_postProcessingInfo.param2.x/20.f);
        float timer = static_cast<float>(i * 3 + m_timer * 40.f);
        float c = 2.0f * (float)cos(timer / 600);
        Vertex center = {0.f, 0.f, 0.f};
        float ratio;
#ifdef USE_KINECT
        Vertex pos;
        ratio = 0.1f;
        if (m_gpuKernel->getSkeletonPosition(i, pos))
        {
            center.x = -pos.x * amplitude.x;
            center.y = pos.y * amplitude.y;
            center.z = pos.z * amplitude.z;
        }
#else
#ifdef USE_LEAPMOTION
        ratio = 1.f;
        if (m_leapMotionController)
        {
            Leap::Frame frame = m_leapMotionController->frame();
            if (frame.isValid())
            {
                Leap::HandList hands = frame.hands();

                int handId(-1);
                Leap::HandList::const_iterator it(hands.begin());
                while (it != hands.end())
                {
                    if (handId == -1)
                    {
                        if (hands.count() != 0)
                        {
                            Leap::Hand hand = frame.hand((*it).id());
                            Leap::FingerList fingers = hand.fingers();
                            if (fingers.count() != 0)
                            {
                                Leap::Finger finger = fingers[i % numMetaballs];
                                Leap::Vector v = finger.tipPosition();
                                metaballs[i].position.x = v.x / 20.f;
                                metaballs[i].position.y = v.z / 20.f;
                                metaballs[i].position.z = v.y / 20.f;
                            }
                        }
                    }
                    ++it;
                }
            }
        }
#else
        ratio = 1.f;
#endif // USE_LEAPMOTION
#endif // USE_KINECT

#ifndef USE_LEAPMOTION
        switch (i % 4)
        {
        case 0:
            metaballs[i].position.x = center.x - ratio * amplitude.x * (float)cos(timer / (740 + i * numMetaballs)) - c;
            metaballs[i].position.y = center.z + ratio * amplitude.y * (float)sin(timer / (620 + i * numMetaballs)) - c;
            metaballs[i].position.z = center.y + fabs(ratio * amplitude.z * (float)sin(cos(timer / (500 + i))) - c);
            break;
        case 1:
            metaballs[i].position.x = center.x + ratio * amplitude.x * (float)sin(timer / (420 + i * numMetaballs)) + c;
            metaballs[i].position.y = center.z - ratio * amplitude.y * (float)cos(timer / (340 + i * numMetaballs)) - c;
            metaballs[i].position.z = center.y + fabs(ratio * amplitude.z * (float)sin(cos(timer / (400 + i))) - c);
            break;
        case 2:
            metaballs[i].position.x = center.x + ratio * amplitude.x * (float)cos(timer / (820 + numMetaballs)) -
                                      0.2f * (float)sin(timer / 600);
            metaballs[i].position.y = center.z + ratio * amplitude.y * (float)sin(timer / (512 + numMetaballs)) -
                                      0.2f * (float)sin(timer / 400);
            metaballs[i].position.z = center.y + fabs(ratio * amplitude.z * (float)sin(cos(timer / (450 + i))) - c);
            break;
        case 3:
            metaballs[i].position.x = center.x + ratio * amplitude.x * (float)cos(timer / (1200 + numMetaballs)) -
                                      0.4f * (float)sin(timer / 1250);
            metaballs[i].position.y = center.z + ratio * amplitude.y * (float)sin(timer / (2120 + numMetaballs)) -
                                      0.4f * (float)sin(timer / 640);
            metaballs[i].position.z = center.y + fabs(ratio * amplitude.z * (float)sin(cos(timer / (450 + i))) - c);
            break;
        }
#endif // USE_LEAPMOTION
    }

    Init();

    // clear the field
    for (int i = 0; i < numVertices; i++)
    {
        vertices[i].value = 0.0f;
        vertices[i].normal.x = 0.f;
        vertices[i].normal.y = 0.f;
        vertices[i].normal.z = 0.f;
    }

    // evaluate the scalar field at each point
    Vertex ballToPoint;
    float squaredRadius;
    Vertex ballPosition;
    float normalScale;
    int i = 0;

#pragma omp parallel
    for (i = 0; i < numMetaballs; i++)
    {
        squaredRadius = metaballs[i].squaredRadius / 4.f;
        ballPosition = metaballs[i].position;

        // VC++6 standard does not inline functions
        // by inlining these maually, in this performance-critical area,
        // almost a 100% increase in speed is found
        for (int j = 0; j < numVertices; j++)
        {
            ballToPoint.x = vertices[j].position.x - ballPosition.x;
            ballToPoint.y = vertices[j].position.y - ballPosition.y;
            ballToPoint.z = vertices[j].position.z - ballPosition.z;

            // get squared distance from ball to point
            // float squaredDistance=ballToPoint.GetSquaredLength();
            float squaredDistance =
                ballToPoint.x * ballToPoint.x + ballToPoint.y * ballToPoint.y + ballToPoint.z * ballToPoint.z;
            if (squaredDistance == 0.0f)
                squaredDistance = 0.0001f;

            // value = r^2/d^2
            vertices[j].value += squaredRadius / squaredDistance;

            // normal = (r^2 * v)/d^4
            normalScale = squaredRadius / (squaredDistance * squaredDistance);
            vertices[j].normal.x += ballToPoint.x * normalScale;
            vertices[j].normal.y += ballToPoint.y * normalScale;
            vertices[j].normal.z += ballToPoint.z * normalScale;
        }
    }

    numFacesDrawn = 0;

    static SURFACE_VERTEX edgeVertices[12];
    {
        // loop through cubes
        int i = 0;
#pragma omp parallel
        for (i = 0; i < numCubes; i++)
        {
            // calculate which vertices are inside the surface
            unsigned char cubeIndex = 0;

            if (cubes[i].vertices[0]->value < threshold)
                cubeIndex |= 1;
            if (cubes[i].vertices[1]->value < threshold)
                cubeIndex |= 2;
            if (cubes[i].vertices[2]->value < threshold)
                cubeIndex |= 4;
            if (cubes[i].vertices[3]->value < threshold)
                cubeIndex |= 8;
            if (cubes[i].vertices[4]->value < threshold)
                cubeIndex |= 16;
            if (cubes[i].vertices[5]->value < threshold)
                cubeIndex |= 32;
            if (cubes[i].vertices[6]->value < threshold)
                cubeIndex |= 64;
            if (cubes[i].vertices[7]->value < threshold)
                cubeIndex |= 128;

            // look this value up in the edge table to see which edges to interpolate along
            int usedEdges = edgeTable[cubeIndex];

            // if the cube is entirely within/outside surface, no faces
            if (usedEdges == 0 || usedEdges == 255)
                continue;

            // update these edges
            for (int currentEdge = 0; currentEdge < 12; currentEdge++)
            {
                if (usedEdges & 1 << currentEdge)
                {
                    CUBE_GRID_VERTEX* v1 = cubes[i].vertices[verticesAtEndsOfEdges[currentEdge * 2]];
                    CUBE_GRID_VERTEX* v2 = cubes[i].vertices[verticesAtEndsOfEdges[currentEdge * 2 + 1]];

                    float delta = (threshold - v1->value) / (v2->value - v1->value);

                    edgeVertices[currentEdge].position.x = v1->position.x + delta * (v2->position.x - v1->position.x);
                    edgeVertices[currentEdge].position.y = v1->position.y + delta * (v2->position.y - v1->position.y);
                    edgeVertices[currentEdge].position.z = v1->position.z + delta * (v2->position.z - v1->position.z);

                    edgeVertices[currentEdge].normal.x = v1->normal.x + delta * (v2->normal.x - v1->normal.x);
                    edgeVertices[currentEdge].normal.y = v1->normal.y + delta * (v2->normal.y - v1->normal.y);
                    edgeVertices[currentEdge].normal.z = v1->normal.z + delta * (v2->normal.z - v1->normal.z);
                }
            }

// send the vertices
#ifdef USE_KINECT
            Vertex center = {0.f, -3000.f, 8000.f};
#else
            Vertex center = {0.f, 0.f, -2500.f};
#endif // USE_KINECT
            for (int k = 0; triTable[cubeIndex][k] != -1; k += 3)
            {
                Vertex n, v, t;

                glBegin(GL_TRIANGLES);
                m_gpuKernel->setCurrentMaterial(1000);

                {
                    n = edgeVertices[triTable[cubeIndex][k]].normal;
                    v = edgeVertices[triTable[cubeIndex][k]].position;
                    t = edgeVertices[triTable[cubeIndex][k]].texCoords;
                    glNormal3f(n.x, n.z, n.y);
                    glTexCoord2f((v.x / minGridSize) + 1.5f, (v.z / minGridSize) + 1.5f);
                    // glTexCoord2f(t.x,t.y);
                    glVertex3f(center.x + scale.x * v.x, center.z + scale.z * v.z, center.y + scale.x * v.y);
                }

                {
                    n = edgeVertices[triTable[cubeIndex][k + 1]].normal;
                    v = edgeVertices[triTable[cubeIndex][k + 1]].position;
                    t = edgeVertices[triTable[cubeIndex][k + 1]].texCoords;
                    glNormal3f(n.x, n.z, n.y);
                    glTexCoord2f((v.x / minGridSize) + 1.5f, (v.z / minGridSize) + 1.5f);
                    // glTexCoord2f(t.x,t.y);
                    glVertex3f(center.x + scale.x * v.x, center.z + scale.z * v.z, center.y + scale.x * v.y);
                }

                {
                    n = edgeVertices[triTable[cubeIndex][k + 2]].normal;
                    v = edgeVertices[triTable[cubeIndex][k + 2]].position;
                    t = edgeVertices[triTable[cubeIndex][k + 2]].texCoords;
                    glNormal3f(n.x, n.z, n.y);
                    glTexCoord2f((v.x / minGridSize) + 1.5f, (v.z / minGridSize) + 1.5f);
                    // glTexCoord2f(t.x,t.y);
                    glVertex3f(center.x + scale.x * v.x, center.z + scale.z * v.z, center.y + scale.x * v.y);
                }

                glEnd();

                numFacesDrawn++;
            }
        }
    }

    addCornellBox(m_cornellBoxType);

    /*
   {
   Vertex r={size.x*scale.x*0.6f,size.y*scale.y*0.4f,size.z*scale.z*0.6f};
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptXYPlane); m_gpuKernel->setPrimitive(m_nbPrimitives,0.f,-4000.f,
   r.z,r.x,r.y,0.f,m_gpuKernel->getCurrentMaterial()+2);
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptXYPlane);
   m_gpuKernel->setPrimitive(m_nbPrimitives,0.f,-4000.f,-r.z,r.x,r.y,0.f,m_gpuKernel->getCurrentMaterial()+2);
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptYZPlane); m_gpuKernel->setPrimitive(m_nbPrimitives,
   r.x,-4000.f,0.f,0.f,r.y,r.z,m_gpuKernel->getCurrentMaterial()+2);
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptYZPlane);
   m_gpuKernel->setPrimitive(m_nbPrimitives,-r.x,-4000.f,0.f,0.f,r.y,r.z,m_gpuKernel->getCurrentMaterial()+2);
   }

   {
   Vertex r={size.x*scale.x*0.62f,size.y*scale.y*0.4f,size.z*scale.z*0.62f};
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptXYPlane); m_gpuKernel->setPrimitive(m_nbPrimitives,0.f,-4000.f,
   r.z,r.x,r.y,0.f,m_gpuKernel->getCurrentMaterial()+2);
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptXYPlane);
   m_gpuKernel->setPrimitive(m_nbPrimitives,0.f,-4000.f,-r.z,r.x,r.y,0.f,m_gpuKernel->getCurrentMaterial()+2);
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptYZPlane); m_gpuKernel->setPrimitive(m_nbPrimitives,
   r.x,-4000.f,0.f,0.f,r.y,r.z,m_gpuKernel->getCurrentMaterial()+2);
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptYZPlane);
   m_gpuKernel->setPrimitive(m_nbPrimitives,-r.x,-4000.f,0.f,0.f,r.y,r.z,m_gpuKernel->getCurrentMaterial()+2);
   }

   {
   Vertex r={size.x*scale.x*0.6f,size.y*scale.y*0.4f,size.z*scale.z*0.6f};
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptXZPlane); m_gpuKernel->setPrimitive(m_nbPrimitives,
   r.x,-4000.f+r.y,0.f,size.x*scale.x*0.02f,0.f,r.z*1.03f,m_gpuKernel->getCurrentMaterial()+2);
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptXZPlane);
   m_gpuKernel->setPrimitive(m_nbPrimitives,-r.x,-4000.f+r.y,0.f,size.x*scale.x*0.02f,0.f,r.z*1.03f,m_gpuKernel->getCurrentMaterial()+2);
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptXZPlane); m_gpuKernel->setPrimitive(m_nbPrimitives, 0.f,-4000.f+r.y,
   r.z,r.x,0.f,size.z*scale.z*0.02f,m_gpuKernel->getCurrentMaterial()+2);
   m_nbPrimitives = m_gpuKernel->addPrimitive(ptXZPlane); m_gpuKernel->setPrimitive(m_nbPrimitives,
   0.f,-4000.f+r.y,-r.z,r.x,0.f,size.z*scale.z*0.02f,m_gpuKernel->getCurrentMaterial()+2);
   }
   */
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
    m_gpuKernel->setPrimitive(m_nbPrimitives, 10000.f, 1000.f, 10000.f, 3000.f, 0.f, 0.f,
                              m_gpuKernel->getCurrentMaterial() + 1);

// Lamp
#ifdef USE_KINECT
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
    m_gpuKernel->setPrimitive(m_nbPrimitives, -8000.f, 8000.f, -8000.f, 500.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL);
    m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives, false);
#else
    // m_nbPrimitives = m_gpuKernel->addPrimitive(  ptSphere ); m_gpuKernel->setPrimitive(  m_nbPrimitives,
    // 18000.f*cos(m_timer/50.f), 10000.f+5000.f*cos(m_timer/80.f), -18000.f*sin(m_timer/120.f), 500.f, 0.f, 0.f,
    // DEFAULT_LIGHT_MATERIAL); m_gpuKernel->setPrimitiveIsMovable(  m_nbPrimitives, false );
    m_nbPrimitives = m_gpuKernel->addPrimitive(ptSphere);
    m_gpuKernel->setPrimitive(m_nbPrimitives, -10000.f, 10000.f, -10000.f, 500.f, 0.f, 0.f, DEFAULT_LIGHT_MATERIAL);
    m_gpuKernel->setPrimitiveIsMovable(m_nbPrimitives, false);
#endif // USE_KINECT

    m_gpuKernel->compactBoxes(true);
    m_timer += 1.5f;
    m_gpuKernel->getSceneInfo().misc.y += 1;
}

void MetaballsScene::doAddLights()
{
    // light
}

#ifdef USE_LEAPMOTION
void MetaballsScene::initializeLeapMotion()
{
    if (m_leapMotionController == nullptr)
    {
        m_leapMotionController = new Leap::Controller();
    }
}

void MetaballsScene::finalizeLeapMotion()
{
    if (m_leapMotionController)
    {
        delete m_leapMotionController;
        m_leapMotionController = nullptr;
    }
}
#endif // USE_LEAPMOTION
