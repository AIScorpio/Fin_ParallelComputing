// #define n_Dim %d
// #define n_PATH %d
// #define n_PERIOD %d
// #define n_Fish %d

// 编译时通过-DVEC_SIZE=4传入向量宽度, 默认float4
#ifndef VEC_SIZE
#define VEC_SIZE 4
#endif

// 动态定义向量类型和转换函数
#if VEC_SIZE == 1
    typedef float float_vec;
    typedef int int_vec;
    #define convert_float_vec(x) ((float)(x))
    #define SUM_VEC(v) (v)
#elif VEC_SIZE == 2
    typedef float2 float_vec;
    typedef int2 int_vec;
    #define convert_float_vec convert_float2
    #define SUM_VEC(v) (v.s0 + v.s1)
#elif VEC_SIZE == 4
    typedef float4 float_vec;
    typedef int4 int_vec;
    #define convert_float_vec convert_float4
    #define SUM_VEC(v) (v.s0 + v.s1 + v.s2 + v.s3)
#elif VEC_SIZE == 8
    typedef float8 float_vec;
    typedef int8 int_vec;
    #define convert_float_vec convert_float8
    #define SUM_VEC(v) (v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7)
#elif VEC_SIZE == 16
    typedef float16 float_vec;
    typedef int16 int_vec;
    #define convert_float_vec convert_float16
    #define SUM_VEC(v) (v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7 + \
                       v.s8 + v.s9 + v.sA + v.sB + v.sC + v.sD + v.sE + v.sF)
#else
    #error "Unsupported VEC_SIZE"
#endif

// **2026-3-29** 
#define n_Vec (n_Dim / VEC_SIZE)         // **2026-4-7** changed from n_Dim / 4 to n_Dim / VEC_SIZE for experiment
#define n_VecPath ((n_PATH + VEC_SIZE - 1) / VEC_SIZE)
// **2026-3-29** 

/* update nParticle positions & velocity, each thread handles one particle */
__kernel void pso_vec(
    // 1. searchGrid
    __global float *position,                // [nDim, nFish]
    __global float *velocity,                // [nDim, nFish]
    __global float *pbest_pos,               // [nDim, nFish]
    __global const float *gbest_pos,         // [nDim]
    __global const float *r1,                // [nDim, nFish]
    __global const float *r2,                // [nDim, nFish]
    const float w, 
    const float c1, 
    const float c2, 
    // 2. American option - fitness each fish
    __global const float_vec *St_vec,            // [nDim, nVecPath]    向量化后的St布局  [nPeriod, nVecPath] 
    const float r, 
    const float T, 
    const float K, 
    const char opt,
    // 3. update pbest
    __global float *pbest_costs              // [nFish]
){
    int gid = get_global_id(0);              // index of the fish
    int nParticle = get_global_size(0);      // nFish

    /* 1. searchGrid */
    #pragma unroll 32/VEC_SIZE
    for (int i = 0; i < n_Dim; i += 4) {          // **2026-3-29** 
        // Vector index starts at (i * nFish + gid)
        int base_idx = i * nParticle + gid;

        // Manually load float4 from 1D arrays
        float4 pos, vel, pbest, r1val, r2val, gbest;

        // Gather float4 from strided memory
        pos.s0 = position[base_idx + 0 * nParticle];
        pos.s1 = position[base_idx + 1 * nParticle];
        pos.s2 = position[base_idx + 2 * nParticle];
        pos.s3 = position[base_idx + 3 * nParticle];

        vel.s0 = velocity[base_idx + 0 * nParticle];
        vel.s1 = velocity[base_idx + 1 * nParticle];
        vel.s2 = velocity[base_idx + 2 * nParticle];
        vel.s3 = velocity[base_idx + 3 * nParticle];

        pbest.s0 = pbest_pos[base_idx + 0 * nParticle];
        pbest.s1 = pbest_pos[base_idx + 1 * nParticle];
        pbest.s2 = pbest_pos[base_idx + 2 * nParticle];
        pbest.s3 = pbest_pos[base_idx + 3 * nParticle];

        r1val.s0 = r1[base_idx + 0 * nParticle];
        r1val.s1 = r1[base_idx + 1 * nParticle];
        r1val.s2 = r1[base_idx + 2 * nParticle];
        r1val.s3 = r1[base_idx + 3 * nParticle];

        r2val.s0 = r2[base_idx + 0 * nParticle];
        r2val.s1 = r2[base_idx + 1 * nParticle];
        r2val.s2 = r2[base_idx + 2 * nParticle];
        r2val.s3 = r2[base_idx + 3 * nParticle];

        gbest.s0 = gbest_pos[i + 0];
        gbest.s1 = gbest_pos[i + 1];
        gbest.s2 = gbest_pos[i + 2];
        gbest.s3 = gbest_pos[i + 3];

        // PSO velocity and position update
        vel = w * vel + c1 * r1val * (pbest - pos) + c2 * r2val * (gbest - pos);
        pos += vel;

        // Scatter float4 back to strided memory
        position[base_idx + 0 * nParticle] = pos.s0;
        position[base_idx + 1 * nParticle] = pos.s1;
        position[base_idx + 2 * nParticle] = pos.s2;
        position[base_idx + 3 * nParticle] = pos.s3;

        velocity[base_idx + 0 * nParticle] = vel.s0;
        velocity[base_idx + 1 * nParticle] = vel.s1;
        velocity[base_idx + 2 * nParticle] = vel.s2;
        velocity[base_idx + 3 * nParticle] = vel.s3;
    }
    // barrier(CLK_GLOBAL_MEM_FENCE);

    /* 2. fitness calculation - American option */
    float dt = T / n_PERIOD;
    float tmp_cost = 0.0f;

    // // **2026-3-29** Private memory cache for pso
    // float pso_private[n_PERIOD];
    // #pragma unroll 8
    // for (int i = 0; i < n_PERIOD; i++) {
    //     pso_private[i] = position[gid + i * nParticle];
    // }
    // // **2026-3-29**

    // 每个线程处理VEC_SIZE个路径
    for (int vec_path=0; vec_path<n_VecPath; vec_path++){
        int_vec bound_idx = (int_vec)(n_PERIOD - 1);            // init to last period
        int St_T_idx = (n_PERIOD-1) * n_VecPath + vec_path;
        float_vec early_excise = St_vec[St_T_idx];               // init to St path_i last period price

        #pragma unroll 8 //VEC_SIZE
        for (int prd=n_PERIOD-1; prd>-1; prd--){
            float cur_fish_val = position[gid + prd * nParticle];     
            // float cur_fish_val = pso_private[prd];  // **2026-3-29** 从私有内存读取，减少全局内存访问
            float_vec cur_St_val = St_vec[vec_path + prd * n_VecPath];

            // 向量化比较更新
            int_vec cmp_mask = isgreaterequal((float_vec)cur_fish_val, cur_St_val);
            bound_idx = select(bound_idx, (int_vec)prd, cmp_mask);               // a>b? a:b will be select(b, a, a>b), mind the sequence!!
            early_excise = select(early_excise, cur_St_val, cmp_mask);

            // // sanity check
            // if (gid == 0 && vec_path == 0){
            //     printf("prd %d | cur_fish_val %.3f | cur_St_val: %.3f %.3f %.3f %.3f | cmp_mask: %d %d %d %d\n", 
            //         prd, cur_fish_val, cur_St_val.s0, cur_St_val.s1, cur_St_val.s2, cur_St_val.s3, 
            //         cmp_mask.s0, cmp_mask.s1, cmp_mask.s2, cmp_mask.s3);
            // }
        }

        // compute current path present value of simulated American option; then cumulate for average later
        // 计算当前向量路径组的期权价值
        float_vec payoffs = max((float_vec)(0.0f), (K - early_excise) * opt);
        float_vec discounts = exp(-r * (convert_float_vec(bound_idx) + 1) * dt);
        float_vec payoff_discount = payoffs * discounts;
        
        // 手动累加4个路径的值到标量tmp_cost
        tmp_cost += SUM_VEC(payoff_discount);
    }

    tmp_cost = tmp_cost / n_PATH;    // get average C_hat for current fish/thread investigation

    /* 3. update pbest */
    if (tmp_cost > pbest_costs[gid]) {
        pbest_costs[gid] = tmp_cost;

        // Copy all dimensions in steps of 4
        #pragma unroll 32/VEC_SIZE
        for (int i = 0; i < n_Dim; i += 4) {            // **2026-3-29** 
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
    }
}


// each thread handle vec_size=4 dimension
__kernel void update_gbest_pos_vec(
    __global float *gbest_pos, 
    __global float *pbest_pos,
    const int gbest_id
){
    int gid = get_global_id(0);     // 0 to nVec_nDim-1
    
    // 每个线程只处理自己的那一组 4 个维度
    int i = gid * 4;
    
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
}