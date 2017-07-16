/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include <vector>
#include "Scene.h"

extern int m_counter;

class SpindizzyScene : public Scene
{

public:
    SpindizzyScene( const std::string& name, const int nbMaxPrimitivePerBox  );
    ~SpindizzyScene(void);

protected:
    virtual void doInitialize();
    virtual void doAnimate();
    virtual void doAddLights();

};
