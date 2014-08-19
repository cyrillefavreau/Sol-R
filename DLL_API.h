/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#pragma warning (disable:4251)

#ifdef WIN32
	#ifdef RAYTRACINGENGINE_EXPORTS
	#define RAYTRACINGENGINE_API __declspec(dllexport)
	#else
	#define RAYTRACINGENGINE_API __declspec(dllimport)
	#endif
#else
	#define RAYTRACINGENGINE_API
#endif
