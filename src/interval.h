#pragma once
#include <cfloat>




class interval {
        public:
            double min, max;
            __device__ __host__ interval() : min(+FLT_MAX), max(-FLT_MAX){} //default interval is empty
            __device__ __host__ interval(double min, double max) : min(min), max(max) {}

            __device__ __host__ 
            double size() const {
                return max - min;
            }

            __device__ __host__ 
            bool contains(double x) const {
                return min <= x && x <= max;
            }
            
            __device__ __host__ 
            bool surrounds(double x) const {
                return min < x && x < max;
            }

            __device__ __host__ 
            double clamp(double x) const {
                if (x < min) return min;
                if (x > max) return max;
                return x;
            }

            static const interval empty, universe;

        private:

    };

   
    