/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#include "Scene.h"

class Year2013 : public Scene
{

public:
    Year2013( const std::string& name, const int nbMaxPrimitivePerBox );
    ~Year2013(void);

protected:
    virtual void doInitialize();
    virtual void doAnimate();
    virtual void doAddLights();

};
