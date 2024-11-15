#pragma once

inline int random_int(int min, int max);
inline double random_double();
inline glm::vec3 unit_vector(const glm::vec3& v);
inline double random_double(float min, float max);

class Perlin {
    public:
        
        Perlin() {
            for (int i = 0; i < point_count; i++) {
                // randFloat[i] = random_double();
                randVec[i] = unit_vector(glm::vec3(random_double(-1, 1), random_double(-1, 1), random_double(-1, 1)));
            }
            perlin_generate_perm(perm_x);
            perlin_generate_perm(perm_y);
            perlin_generate_perm(perm_z);
        }

        __device__ __host__
        float noise(const glm::vec3& p) {
            auto i = static_cast<int>(4 * p.x) & 255;
            auto j = static_cast<int>(4 * p.y) & 255;
            auto k = static_cast<int>(4 * p.z) & 255;

            return randFloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
        }

        __device__ __host__
        float trilinear_noise_smoothing(const glm::vec3& p) {
            auto u = p.x - floorf(p.x);
            auto v = p.y - floorf(p.y);
            auto w = p.z - floorf(p.z);

            auto i = static_cast<int>(floorf(p.x));
            auto j = static_cast<int>(floorf(p.y));
            auto k = static_cast<int>(floorf(p.z));
            float c[2][2][2];

            for (int di = 0; di < 2; di++)
                for(int dj = 0; dj < 2; dj++)
                    for(int dk = 0; dk < 2; dk++)
                        c[di][dj][dk] = randFloat[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255] ];

            return trilinear_interpolation(c, u, v, w);
                
        }

        __device__ __host__
        float hermitian_noise_smoothing(const glm::vec3& p) {
            auto u = p.x - floorf(p.x);
            auto v = p.y - floorf(p.y);
            auto w = p.z - floorf(p.z);

            u = u * u * (3 - 2 * u);
            v = v * v * (3 - 2 * v);
            w = w * w * (3 - 2 * w);

            auto i = static_cast<int>(floorf(p.x));
            auto j = static_cast<int>(floorf(p.y));
            auto k = static_cast<int>(floorf(p.z));
            float c[2][2][2];
    

            for (int di = 0; di < 2; di++)
                for(int dj = 0; dj < 2; dj++)
                    for(int dk = 0; dk < 2; dk++){
                        c[di][dj][dk] = randFloat[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255] ];
                    }

            return trilinear_interpolation(c, u, v, w);
                
        }
        

        __device__ __host__
        float perlin_noise_smoothing(const glm::vec3& p) {
            auto u = p.x - floorf(p.x);
            auto v = p.y - floorf(p.y);
            auto w = p.z - floorf(p.z);

            // u = u * u * (3 - 2 * u);
            // v = v * v * (3 - 2 * v);
            // w = w * w * (3 - 2 * w);

            auto i = static_cast<int>(floorf(p.x));
            auto j = static_cast<int>(floorf(p.y));
            auto k = static_cast<int>(floorf(p.z));
            // float c[2][2][2];
            glm::vec3 c[2][2][2];

            for (int di = 0; di < 2; di++)
                for(int dj = 0; dj < 2; dj++)
                    for(int dk = 0; dk < 2; dk++){
                        // c[di][dj][dk] = randFloat[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255] ];
                        c[di][dj][dk] = randVec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255] ];
                    }

            // return trilinear_interpolation(c, u, v, w);
            return perlin_interpolation(c, u, v, w);
                
        }




    
    private:
        static const int point_count = 256;
        glm::vec3 randVec[point_count];
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

        __device__ __host__
        static float trilinear_interpolation(float c[2][2][2], float u, float v, float w) {
            auto accum = 0.0f;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    for (int k = 0; k < 2; k++)
                        accum += (i * u + (1 - i) * (1 - u))
                               * (j * v + (1 - j) * (1 - v))
                               * (k * w + (1 - k) * (1 - w))
                               * c[i][j][k];
            return accum;
        }
        __device__ __host__
        static float perlin_interpolation(const glm::vec3 c[2][2][2], float u, float v, float w) {
            auto uu = u * u * (3 - 2 * u);
            auto vv = v * v * (3 - 2 * v);
            auto ww = w * w * (3 - 2 * w);

            auto accum = 0.0f;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    for (int k = 0; k < 2; k++) {
                        glm::vec3 weight_v(u - i, v - j, w - k);
                        accum += (i * uu + (1 - i) * (1 - uu))
                               * (j * vv + (1 - j) * (1 - vv))
                               * (k * ww + (1 - k) * (1 - ww))
                               * glm::dot(c[i][j][k], weight_v);
                    }
            return accum;


        }



};