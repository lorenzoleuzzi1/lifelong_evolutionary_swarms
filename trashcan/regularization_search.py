from environment import SwarmForagingEnv, RED, BLUE

if __name__ == "__main__":
    search = "coarse"
    reg = "wp"
    distribution = "b"
    drifts = [RED, BLUE, RED]
    seed = 1

    if search == "coarse":

        if reg == "gd":
            gd_lambda_range_u_coarse = [5, 6, 7, 8, 9, 10, 11]
            gd_lambda_range_b_coarse = [7, 8, 9, 10, 11, 12, 13]

            if distribution == "u":
                param_space = gd_lambda_range_u_coarse
            else:
                param_space = gd_lambda_range_b_coarse

        if reg == "wp":
            wp_lambda_1_u_coarse = [0.001, 0.01, 0.1, 1.0]
            wp_lambda_2_u_coarse = [0.0001, 0.001, 0.01, 0.1]
            grid_wp_lambda_u_coarse = [(l1, l2) for l1 in wp_lambda_1_u_coarse for l2 in wp_lambda_2_u_coarse]
            grid_wp_lambda_u_coarse = [l for l in grid_wp_lambda_u_coarse if l[0] > l[1]] # eliminates values where l2 > l1

            wp_lambda_1_b_coarse = [0.002, 0.02, 0.2, 2.0]
            wp_lambda_2_b_coarse = [0.0002, 0.002, 0.02, 0.2]
            grid_wp_lambda_b_coarse = [(l1, l2) for l1 in wp_lambda_1_b_coarse for l2 in wp_lambda_2_b_coarse]
            grid_wp_lambda_b_coarse = [l for l in grid_wp_lambda_b_coarse if l[0] > l[1]] # eliminates values where l2 > l1

            if distribution == "u":
                param_space = grid_wp_lambda_u_coarse
            else:
                param_space = grid_wp_lambda_b_coarse

    print(param_space)

    
