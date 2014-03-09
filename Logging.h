/* 
* Copyright (C) 2011-2014 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
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
#if 1
	#define LOG_INFO( __level, __msg ) \
   { \
      if( __level==1) std::cout << getTimestamp() << " INFO  [" << __level << "] " << __msg << std::endl; \
   }
	#define LOG_ERROR( __msg ) \
      std::cout << getTimestamp() << " ERROR [1] " << __msg << std::endl;
#else
	#define LOG_INFO( __level, __msg ) \
	   { \
		  if( m_activeLogging ) \
		  { \
			 std::stringstream __s; \
			 __s << "[INFO]  " << __msg; \
			 OutputDebugString(__s.str().c_str()); \
		  } \
	   }

	#define LOG_ERROR( __msg ) \
	   { \
		  std::stringstream __s; \
		  __s << "[ERROR] " << __msg; \
		  OutputDebugString(__s.str().c_str()); \
	   }
#endif // 0
#else
	#define LOG_INFO( __level, __msg )
	#define LOG_ERROR( __msg )
#endif
