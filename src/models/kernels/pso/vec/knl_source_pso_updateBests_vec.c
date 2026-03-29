#define n_Dim %d
#define n_Fish %d

#define n_Dim_main (n_Dim / 4 * 4)       // **2026-3-29** for unrolling, only process multiples of 4 dimensions, the rest will be ignored (if any)

// each thread takes care of one fish
// 1. Compare pBest_costs vs Costs     scaler vs scaler
// 2. if Costs > pBest_costs           scaler
// 3. pBest_costs[i] = Costs[i] & pBest_pos[i] = position[i]              [nDim]

__kernel void update_pbest_f2f4(
    __global float *costs,
    __global float *pbest_costs,
    __global float *position,
    __global float *pbest_pos
){
    int gid = get_global_id(0);             // current fish ID
    int nParticle = get_global_size(0);

    if (costs[gid] > pbest_costs[gid]) {
        pbest_costs[gid] = costs[gid];

        // Copy all dimensions in steps of 4
        #pragma unroll 8
        for (int i = 0; i < n_Dim; i += 4) {      // **2026-3-29**
        // for (int i = 0; i < n_Dim_main; i += 4) {    // **2026-3-29** unroll 8 means process 32 dimensions per thread, so loop step is 4 (float4)
            int base_idx = i * nParticle + gid;

            float4 pos_vec = (float4)(
                position[base_idx + 0 * nParticle],
                position[base_idx + 1 * nParticle],
                position[base_idx + 2 * nParticle],
                position[base_idx + 3 * nParticle]
            );

            pbest_pos[base_idx + 0 * nParticle] = pos_vec.s0;
            pbest_pos[base_idx + 1 * nParticle] = pos_vec.s1;
            pbest_pos[base_idx + 2 * nParticle] = pos_vec.s2;
            pbest_pos[base_idx + 3 * nParticle] = pos_vec.s3;
        }
        // // **2026-3-29**
        // for (int i = n_Dim_main; i < n_Dim; i++){
        //     int idx = i * nParticle + gid;
        //     pbest_pos[idx] = position[idx];
        // }
        // // **2026-3-29**
    }
}


// each thread handle vec_size=4 dimension
__kernel void update_gbest_pos_f2f4(
    __global float *gbest_pos, 
    __global float *pbest_pos,
    const int gbest_id
){
    int gid = get_global_id(0);     // 0 to nVec_nDim-1, each thread handles one float4 (4 dimensions)

    int i = gid * 4;                // 直接计算该线程负责的维度起始

    int idx0 = (i + 0) * n_Fish + gbest_id;
    int idx1 = (i + 1) * n_Fish + gbest_id;
    int idx2 = (i + 2) * n_Fish + gbest_id;
    int idx3 = (i + 3) * n_Fish + gbest_id;
    float4 val = (float4)(
        pbest_pos[idx0],
        pbest_pos[idx1],
        pbest_pos[idx2],
        pbest_pos[idx3]
    );
    gbest_pos[i + 0] = val.s0;
    gbest_pos[i + 1] = val.s1;
    gbest_pos[i + 2] = val.s2;
    gbest_pos[i + 3] = val.s3;
    // 没有 for 循环！每个线程只做一次

    // #pragma unroll 8
    // for (int i = 0; i < n_Dim; i += 4) {         // **2026-3-29**
    // // for (int i = 0; i < n_Dim_main; i += 4) {       // **2026-3-29** unroll 8 means process 32 dimensions per thread, so loop step is 4 (float4)
    //     int idx0 = (i + 0) * n_Fish + gbest_id;
    //     int idx1 = (i + 1) * n_Fish + gbest_id;
    //     int idx2 = (i + 2) * n_Fish + gbest_id;
    //     int idx3 = (i + 3) * n_Fish + gbest_id;

    //     float4 val = (float4)(
    //         pbest_pos[idx0],
    //         pbest_pos[idx1],
    //         pbest_pos[idx2],
    //         pbest_pos[idx3]
    //     );

    //     gbest_pos[i + 0] = val.s0;
    //     gbest_pos[i + 1] = val.s1;
    //     gbest_pos[i + 2] = val.s2;
    //     gbest_pos[i + 3] = val.s3;
    // }
    // // **2026-3-29**
    // for (int i = n_Dim_main; i < n_Dim; i++){
    //     gbest_pos[i] = pbest_pos[i * n_Fish + gbest_id];
    // }
    // // **2026-3-29**
}