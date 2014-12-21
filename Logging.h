/* 
 * Copyright (C) 2014 Cyrille Favreau - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Cyrille Favreau <cyrille_favreau@hotmail.com>
 */

#pragma once

#ifdef WIN32
	#ifdef LOGGING
		#include <ETWLoggingModule.h>
		#include <ETWResources.h>
	#else
		#include <sstream>
	#endif
	#include <windows.h>
#else
#include <iostream>
#include <string>
#endif // WIN32

#include <time.h>

inline std::string getTimestamp()
{
#ifdef WIN32
  char buffer[80] = {'\0'} ;    
  time_t now = time( &now ) ;
  struct tm local_time; 
  localtime_s( &local_time, &now ) ;
  strftime( buffer, BUFSIZ, "%m/%d/%Y %H:%M:%S", &local_time ) ;    
  return std::string(buffer);
#else
  return "";
#endif // WIN32
}

#ifdef WIN32
#if 0
	#define LOG_INFO( __level, __msg ) \
   { \
      if( __level==1) std::cout << getTimestamp() << " INFO  [" << __level << "] " << __msg << std::endl; \
   }
	#define LOG_ERROR( __msg ) \
      std::cout << getTimestamp() << " ERROR [1] " << __msg << std::endl;
#else
	#define LOG_INFO( __level, __msg ) \
	   { \
         if( __level==1) {\
			   std::stringstream __s; \
            __s << getTimestamp() << " [" << GetCurrentThreadId() << "] [INFO]  " << __msg; \
   			OutputDebugString(__s.str().c_str()); \
         }\
	   }

	#define LOG_ERROR( __msg ) \
	   { \
		  std::stringstream __s; \
		  __s << getTimestamp() << " [" << GetCurrentThreadId() << "] [ERROR] " << __msg; \
		  OutputDebugString(__s.str().c_str()); \
	   }
#endif // 0
#else
	#define LOG_INFO( __level, __msg ) \
	{ \
		if( __level==1) std::cout << __msg << std::endl; \
	}
	#define LOG_ERROR( __msg ) std::cerr << __msg << std::endl;
#endif
