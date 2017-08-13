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

#include <scenes/Scene.h>

class GraphScene : public Scene
{
public:
    GraphScene(const std::string& name);
    ~GraphScene(void);

protected:
    virtual void doInitialize();
    virtual void doAnimate();
    virtual void doAddLights();

private:
    void buildGraph(const bool update);
    void buildChart(const std::string& filename);
    void buildFrame();

private:
    float m_values[100][100];
    vec2i m_valueSize;
    vec4f m_graphSize;
    vec4f m_graphScale;
    vec4f m_graphCenter;

    int m_nbGraphElements;
    float* m_graphValues;
    int m_startGraph;
    float m_graphSpace;
    int m_graphMaterial;
    int m_graphObjectsPerBox;
};
