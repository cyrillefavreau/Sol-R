#pragma once

#ifdef WIN32
	#ifdef LOGGING
		#include <ETWLoggingModule.h>
		#include <ETWResources.h>
	#else
		#include <sstream>
	#endif
	#include <windows.h>
#endif // WIN32

#ifdef WIN32
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
#else
	#define LOG_INFO( __level, __msg )
	#define LOG_ERROR( __msg )
#endif
