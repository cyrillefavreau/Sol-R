/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"
class GraphScene :
   public Scene
{

public:
   GraphScene( const std::string& name, const int nbMaxPrimitivePerBox  );
   ~GraphScene(void);

protected:
   virtual void doInitialize();
   virtual void doAnimate();
   virtual void doAddLights();

   virtual void renderText();

private:
   void buildGraph(bool update);
   void buildChart(const std::string& filename);

private:

   float  m_values[100][100];
   INT2   m_valueSize;
   Vertex m_graphSize;
   Vertex m_graphScale;
   Vertex m_graphCenter;

   int    m_nbGraphElements;
   float* m_graphValues;
   int    m_startGraph;
   float  m_graphSpace;
   int    m_graphMaterial;
   int    m_graphMode;
   int    m_graphObjectsPerBox;
};

