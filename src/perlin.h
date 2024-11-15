#pragma once

inline int random_int(int min, int max);
inline double random_double();

class Perlin {
    public:
        
        Perlin() {
            for (int i = 0; i < point_count; i++) {
                randFloat[i] = random_double();
            }
            perlin_generate_perm(perm_x);
            perlin_generate_perm(perm_y);
            perlin_generate_perm(perm_z);
        }

        __device__ __host__
        float noise(const glm::vec3& p) {
            auto i = int(4 * p.x) & 255;
            auto j = int(4 * p.y) & 255;
            auto k = int(4 * p.z) & 255;

            return randFloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
        }


    
    private:
        static const int point_count = 256;
        float randFloat[point_count];
        int perm_x[point_count];
        int perm_y[point_count];
        int perm_z[point_count];

        
        static void perlin_generate_perm(int* p) {
            for (int i = 0; i < point_count; i++)
                p[i] = i;
            permute(p, point_count);
        }

        
        static void permute(int* p, int n) {
            for (int i = n - 1; i > 0; i--) {
                int target = random_int(0, 1);
                int tmp = p[i];
                p[i] = p[target];
                p[target] = tmp;
            }
        }

};