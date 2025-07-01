cmake_minimum_required (VERSION 3.14)

if (FASTBUILD)
  target_sources(CMSISDSP PRIVATE FastMathFunctions/FastMathFunctions.c)

  if ((NOT ARMAC5) AND (NOT DISABLEFLOAT16))
    target_sources(CMSISDSP PRIVATE FastMathFunctions/FastMathFunctionsF16.c)
  endif()

else()

target_sources(CMSISDSP PRIVATE FastMathFunctions/arm_atan2_f32.c)
target_sources(CMSISDSP PRIVATE FastMathFunctions/arm_atan2_q31.c)
target_sources(CMSISDSP PRIVATE FastMathFunctions/arm_atan2_q15.c)


endif()


