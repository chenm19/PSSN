import argparse
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils_log import save_logging, load_logging
from class_AC_TPC import AC_TPC, initialize_embedding
from tools import *
from box_adjusted import draw_boxplt


def train(main_path, opt, data_x, times_count, parameters, base_dic, base_res, base_res_inter, print_flag=True):
    if print_flag:
        print("[{:0>4d}][Step 1] Loading data".format(times_count))
    # data_x = load_data(main_path, "/data/data_x_new.npy")
    data_y = load_data(main_path, "/data/data_y/data_y_{}.npy".format(opt.data[0:-1]))
    seed = 1234
    tr_data_x, te_data_x, tr_data_y, te_data_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=seed
    )
    print("tr_data_y", len(tr_data_y), len(tr_data_y[0]))
    print("tr_data_x", len(tr_data_x), len(tr_data_x[0]))
    tr_data_x, va_data_x, tr_data_y, va_data_y = train_test_split(
        tr_data_x, tr_data_y, test_size=0.2, random_state=seed
    )
    print("va_data_y", len(va_data_y), len(va_data_y[0]))
    print("va_data_x", len(va_data_x), len(va_data_x[0]))
    time.sleep(1)
    y_type = 'continuous'


    if print_flag:
        print("[{:0>4d}][Step 2] Define network parameters".format(times_count))
    # DEFINE NETWORK PARAMETERS
    K = parameters.get("K")  # 5  # 5
    h_dim_FC = parameters.get("h_dim_FC")  # 26  # for fully_connected layers
    h_dim_RNN = parameters.get("h_dim_RNN")  # 26
    x_dim = np.shape(data_x)[2]
    y_dim = np.shape(data_y)[2]
    num_layer_encoder = parameters.get("num_layer_encoder")  # 2
    num_layer_selector = parameters.get("num_layer_selector")  # 3
    num_layer_predictor = parameters.get("num_layer_predictor")  # 2
    z_dim = h_dim_RNN * num_layer_encoder
    max_length = np.shape(data_x)[1]
    rnn_type = 'LSTM'  # GRU, LSTM
    input_dims = {
        'x_dim': x_dim,
        'y_dim': y_dim,
        'y_type': y_type,
        'max_cluster': K,
        'max_length': max_length
    }
    network_settings = {
        'h_dim_encoder': h_dim_RNN,
        'num_layers_encoder': num_layer_encoder,
        'rnn_type': rnn_type,
        'rnn_activate_fn': tf.nn.tanh,
        'h_dim_selector': h_dim_FC,
        'num_layers_selector': num_layer_selector,
        'h_dim_predictor': h_dim_FC,
        'num_layers_predictor': num_layer_predictor,
        'fc_activate_fn': tf.nn.relu
    }

    # TRAIN -- INITIALIZE NETWORK
    if print_flag:
        print("[{:0>4d}][Step 3] Initialize network".format(times_count))
    lr_rate = parameters.get("lr_rate")  # 0.0001  # 0.0001
    keep_prob = parameters.get("keep_prob_s3")  # 0.5  # 0.5
    mb_size = parameters.get("mb_size_s3")  # 32  # 128 # 32
    # times_count = '10'
    ITERATION = parameters.get("iteration_s3")  # 1000  # 3750
    check_step = parameters.get("check_step_s3")  # 250  # 250
    save_path = main_path + 'saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/proposed/init/'.format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count)  # ENZE updated
    if not os.path.exists(save_path + '/models/'):
        os.makedirs(save_path + '/models/')
    tf.reset_default_graph()
    # Turn on xla optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = AC_TPC(sess, "AC_TPC", input_dims, network_settings)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer(), feed_dict={model.E: np.zeros([K, z_dim]).astype(float)})
    avg_loss = 0
    for itr in range(ITERATION):
        x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)
        _, tmp_loss = model.train_mle(x_mb, y_mb, lr_rate, keep_prob)
        avg_loss += tmp_loss / check_step
        if (itr + 1) % check_step == 0:
            # print("va_data_x", len(va_data_x), len(va_data_x[0]))
            # tmp_y, tmp_m = model.predict_y_hats(va_data_x)
            tmp_y, tmp_m = model.predict_y_hats(va_data_x)
            # print("y_dim =", y_dim)
            # print("tmp_y", len(tmp_y), len(tmp_y[0]))
            # print("tmp_m", len(tmp_m), len(tmp_m[0]))
            y_pred = tmp_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
            # print("va_data_y", len(va_data_y), len(va_data_y[0]))
            # print("tmp_m", len(tmp_m), len(tmp_m[0]))
            # time.sleep(3)
            y_true = va_data_y.reshape([-1, y_dim])[tmp_m.reshape([-1]) == 1]
            val_loss = np.sum((y_true - y_pred) ** 2, axis=-1)
            avg_val_loss = np.mean(val_loss)
            print("ITR {:05d}: loss_train={:.3f} loss_val={:.3f}".format(itr + 1, avg_loss, avg_val_loss))
            avg_loss = 0
    saver.save(sess, save_path + 'models/model_K{}'.format(K))
    save_logging(network_settings, save_path + 'models/network_settings_K{}.txt'.format(K))

    # TRAIN TEMPORAL PHENOTYPING
    if print_flag:
        print("[{:0>4d}][Step 4] Train temporal phenotyping".format(times_count))
    alpha = parameters.get("alpha")  # 0.001  # 0.00001
    beta = parameters.get("beta")  # 1  # 1
    mb_size = parameters.get("mb_size_s4")  # 8  # 128#8
    M = int(tr_data_x.shape[0] / mb_size)  # for main algorithm
    keep_prob = parameters.get("keep_prob_s4")  # 0.7
    lr_rate1 = parameters.get("lr_rate1")  # 0.0001  # 0.0001
    lr_rate2 = parameters.get("lr_rate2")  # 0.0001  # 0.0001
    save_path = main_path + 'saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/proposed/trained/'.format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count)  # ENZE updated
    if not os.path.exists(save_path + '/models/'):
        os.makedirs(save_path + '/models/')
    if not os.path.exists(save_path + '/results/'):
        os.makedirs(save_path + '/results/')

    # LOAD INITIALIZED NETWORK
    if print_flag:
        print("[{:0>4d}][Step 5] Load initialized network".format(times_count))
    load_path = main_path + 'saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/proposed/init/'.format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count)  # ENZE updated
    tf.reset_default_graph()
    # Turn on xla optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    network_settings = load_logging(load_path + 'models/network_settings_K{}.txt'.format(K))
    z_dim = network_settings['num_layers_encoder'] * network_settings['h_dim_encoder']
    model = AC_TPC(sess, "AC_TPC", input_dims, network_settings)
    saver = tf.train.Saver()
    saver.restore(sess, load_path + 'models/model_K{}'.format(K))

    # INITIALIZING EMBEDDING & SELECTOR
    if print_flag:
        print("[{:0>4d}][Step 6] Initializing embedding & selector".format(times_count))
    # K-means over the latent encodings
    e, s_init, tmp_z = initialize_embedding(model, tr_data_x, K)
    e = np.arctanh(e)
    sess.run(model.EE.initializer, feed_dict={model.E: e})  # model.EE = tf.nn.tanh(model.E)
    # update selector wrt initial classes
    ITERATION = parameters.get("iteration_s6")  # 8000
    check_step = parameters.get("check_step_s6")  # 1000
    avg_loss_s = 0
    for itr in range(ITERATION):
        z_mb, s_mb = f_get_minibatch(mb_size, tmp_z, s_init)
        _, tmp_loss_s = model.train_selector(z_mb, s_mb, lr_rate1, k_prob=keep_prob)
        avg_loss_s += tmp_loss_s / check_step
        if (itr + 1) % check_step == 0:
            print("ITR:{:04d} | Loss_s:{:.4f}".format(itr + 1, avg_loss_s))
            avg_loss_s = 0
    tmp_ybars = model.predict_yy(np.tanh(e))
    new_e = np.copy(e)

    if print_flag:
        print("[{:0>4d}][Step 7] Training main algorithm".format(times_count))
    # TRAINING MAIN ALGORITHM
    ITERATION = parameters.get("iteration_s7")  # 500  # 5000
    check_step = parameters.get("check_step_s7")  # 100
    avg_loss_c_L1 = 0
    avg_loss_a_L1 = 0
    avg_loss_a_L2 = 0
    avg_loss_e_L1 = 0
    avg_loss_e_L3 = 0
    va_avg_loss_L1 = 0
    va_avg_loss_L2 = 0
    va_avg_loss_L3 = 0
    for itr in range(ITERATION):
        e = np.copy(new_e)
        for _ in range(M):
            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)
            _, tmp_loss_c_L1 = model.train_critic(x_mb, y_mb, lr_rate1, keep_prob)
            avg_loss_c_L1 += tmp_loss_c_L1 / (M * check_step)
            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)
            _, tmp_loss_a_L1, tmp_loss_a_L2 = model.train_actor(x_mb, y_mb, alpha, lr_rate2, keep_prob)
            avg_loss_a_L1 += tmp_loss_a_L1 / (M * check_step)
            avg_loss_a_L2 += tmp_loss_a_L2 / (M * check_step)
        for _ in range(M):
            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_x, tr_data_y)
            _, tmp_loss_e_L1, tmp_loss_e_L3 = model.train_embedding(x_mb, y_mb, beta, lr_rate1, keep_prob)
            avg_loss_e_L1 += tmp_loss_e_L1 / (M * check_step)
            avg_loss_e_L3 += tmp_loss_e_L3 / (M * check_step)
        x_mb, y_mb = f_get_minibatch(mb_size, va_data_x, va_data_y)
        tmp_loss_L1, tmp_loss_L2, tmp_loss_L3 = model.get_losses(x_mb, y_mb)
        va_avg_loss_L1 += tmp_loss_L1 / check_step
        va_avg_loss_L2 += tmp_loss_L2 / check_step
        va_avg_loss_L3 += tmp_loss_L3 / check_step
        new_e = sess.run(model.embeddings)
        if (itr + 1) % check_step == 0:
            tmp_ybars = model.predict_yy(new_e)
            tmp_line = "ITR {:04d}: L1_c={:.3f}  L1_a={:.3f}  L1_e={:.3f}  L2={:.3f}  L3={:.3f} || va_L1={:.3f}  va_L2={:.3f}  va_L3={:.3f}".format(
                itr + 1, avg_loss_c_L1, avg_loss_a_L1, avg_loss_e_L1, avg_loss_a_L2, avg_loss_e_L3,
                va_avg_loss_L1, va_avg_loss_L2, va_avg_loss_L3)
            print(tmp_line)
            avg_loss_c_L1 = 0
            avg_loss_a_L1 = 0
            avg_loss_a_L2 = 0
            avg_loss_e_L1 = 0
            avg_loss_e_L3 = 0
            va_avg_loss_L1 = 0
            va_avg_loss_L2 = 0
            va_avg_loss_L3 = 0
    # print('=============================================')
    # f1.close()

    if print_flag:
        print("[{:0>4d}][Step 8] Saving".format(times_count))
    saver.save(sess, save_path + '/models/model_K{}'.format(K))
    save_logging(network_settings, save_path + '/models/network_settings_K{}.txt'.format(K))
    ##### np.savez(save_path + 'models/embeddings.npz', e=e)
    saver.restore(sess, save_path + 'models/model_K{}'.format(K))

    _, tmp_pi, tmp_m = model.predict_zbars_and_pis_m2(data_x)
    # tmp_pi = tmp_pi.reshape([-1, K])[tmp_m.reshape([-1]) == 1]
    # print(tmp_pi)
    # ncol = nrow = int(np.ceil(np.sqrt(K)))
    # plt.figure(figsize=[4 * ncol, 2 * nrow])
    # for k in range(K):
    #     plt.subplot(ncol, nrow, k + 1)
    #     plt.hist(tmp_pi[:, k])
    # plt.suptitle("Clustering assignment probabilities")
    # plt.show()
    # # plt.savefig(save_path + 'results/figure_clustering_assignments.png')
    # plt.close()
    pred_y, tmp_m = model.predict_s_sample(data_x)
    pred_y = pred_y.reshape([-1, 1])[tmp_m.reshape([-1]) == 1]
    # print(np.unique(pred_y))
    # plt.hist(pred_y[:, 0], bins=15, color='C1', alpha=1.0)
    # plt.show()
    # plt.savefig(save_path + 'results/figure_clustering_hist.png')
    # plt.close()
    # patientProgressions = np.array_split(pred_y, 320)
    pred_y = [item[0] for item in pred_y]
    pt_dic = load_patient_dictionary(main_path, opt.data[:-1])
    pt_ids = np.load(main_path + "data/ptid.npy", allow_pickle=True)
    output_labels = []
    tmp_index = 0
    for pt_id in pt_ids:
        output_labels.append(pred_y[tmp_index: tmp_index + len(pt_dic.get(pt_id))])
        tmp_index += len(pt_dic.get(pt_id))
    # print(patientProgressions)
    np.save(main_path + "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/labels_{}.npy".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, times_count), output_labels, allow_pickle=True)
    with open(main_path + "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/parameters_{}.txt".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, times_count), "w") as f:
        string = "# [Step 2] Define network parameters\n" \
                 "'K': {},\n" \
                 "'h_dim_FC': {},\n" \
                 "'h_dim_RNN': {},\n" \
                 "'num_layer_encoder': {},\n" \
                 "'num_layer_selector': {},\n" \
                 "'num_layer_predictor': {},\n" \
                 "# [Step 3] Initialize network\n" \
                 "'lr_rate': {},\n" \
                 "'keep_prob_s3': {},\n"\
                 "'mb_size_s3': {},\n" \
                 "'iteration_s3': {},\n" \
                 "'check_step_s3': {},\n" \
                 "# [Step 4] Train temporal phenotyping\n" \
                 "'alpha': {},\n" \
                 "'beta': {},\n" \
                 "'mb_size_s4': {},\n" \
                 "'keep_prob_s4': {},\n" \
                 "'lr_rate1': {},\n" \
                 "'lr_rate2': {},\n" \
                 "# [Step 6] Initializing embedding & selector\n" \
                 "'iteration_s6': {},\n" \
                 "'check_step_s6': {},\n" \
                 "# [Step 7] Training main algorithm\n" \
                 "'iteration_s7': {},\n" \
                 "'check_step_s7': {}\n".format(
            # [Step 2] Define network parameters
            parameters.get("K"),
            parameters.get("h_dim_FC"),
            parameters.get("h_dim_RNN"),
            parameters.get("num_layer_encoder"),
            parameters.get("num_layer_selector"),
            parameters.get("num_layer_predictor"),
            # [Step 3] Initialize network
            parameters.get("lr_rate"),
            parameters.get("keep_prob_s3"),
            parameters.get("mb_size_s3"),
            parameters.get("iteration_s3"),
            parameters.get("check_step_s3"),
            # [Step 4] Train temporal phenotyping
            parameters.get("alpha"),
            parameters.get("beta"),
            parameters.get("mb_size_s4"),
            parameters.get("keep_prob_s4"),
            parameters.get("lr_rate1"),
            parameters.get("lr_rate2"),
            # [Step 6] Initializing embedding & selector
            parameters.get("iteration_s6"),
            parameters.get("check_step_s6"),
            # [Step 7] Training main algorithm
            parameters.get("iteration_s7"),
            parameters.get("check_step_s7")
        )
        f.write(string)
    # print(output_labels)
    print("Labels ready (0/6)")
    heat_map_data = get_heat_map_data(main_path, int(opt.k), output_labels, opt.data[:-1])
    _, heat_map_data_inter_14 = get_heat_map_data_inter(main_path, int(opt.k), output_labels, opt.data[:-1], False)
    _, heat_map_data_inter_2 = get_heat_map_data_inter(main_path, int(opt.k), output_labels, opt.data[:-1], True)
    sustain_intra = get_static_sustain(main_path, "intra")
    sustain_inter = get_static_sustain(main_path, "inter")
    print("Heat map data are built (1/6)")

    draw_heat_map_3(base_res, sustain_intra, heat_map_data, "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/intra_cluster_id={}_cols=7".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, times_count), 2, False, 7)
    draw_heat_map_3(base_res, sustain_intra, heat_map_data, "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/intra_cluster_id={}_cols=14".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, times_count), 2, False, 14)
    print("Intra built (2/6)")
    try:
        draw_stairs_3(base_res_inter, sustain_inter, heat_map_data_inter_14, main_path + "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/inter_cluster_id={}".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, times_count))
    except Exception as err:
        print("Failed in drawing stairs 14:", err)
    try:
        draw_stairs_3(base_res_inter, sustain_inter, heat_map_data_inter_2, main_path + "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/modified_inter_cluster_id={}".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, times_count))
    except Exception as err:
        print("Failed in drawing stairs 2:", err)
    print("Inter built (3/6)")

    box_data_save_path = "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/dist/box_data_{}_k={}_id={}.pkl".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, opt.data, int(opt.k), times_count)
    box_data_dist_save_path = "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/dist/".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, opt.data, int(opt.k), times_count)
    if not os.path.exists(main_path + box_data_dist_save_path):
        os.makedirs(main_path + box_data_dist_save_path)

    make_heat_map_data_box(main_path, box_data_save_path, output_labels, opt.data[:-1])
    draw_boxplt(main_path + box_data_save_path, main_path + box_data_dist_save_path, opt.data, int(opt.k), times_count)
    print("Hist built (4/6)")

    triangle_labels_x = get_triangle_data_x(main_path, output_labels, opt.data, int(opt.k))
    triangle_labels_y = get_triangle_data_y(main_path, output_labels, opt.data, int(opt.k))
    draw_triangle(triangle_labels_x, "data_x", "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/triangle_{}_k={}_id={}_data_x.png".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, opt.data, int(opt.k), times_count))
    draw_triangle(triangle_labels_y, "data_y", "saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/triangle_{}_k={}_id={}_data_y.png".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count, opt.data, int(opt.k), times_count))
    print("Triangle built (5/6)")

    # print("heat_map_data_inter in train:")
    # print(heat_map_data_inter)
    judge, judge_params, distribution_string = judge_good_train(output_labels, opt.data[:-1], heat_map_data, heat_map_data_inter_14, True, base_dic, int(opt.k))

    if int(opt.clear) == 1:
        shutil.rmtree("saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/proposed".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count))
        print("Removed folder {} after training (6/6)".format("saves/data={}_alpha={}_beta={}_h_dim={}_main_epoch={}/{}/proposed".format(opt.data, float(opt.alpha), float(opt.beta), int(opt.h_dim), int(opt.main_epoch), times_count)))
    return judge, judge_params, distribution_string


def start(params, opt):
    main_path = os.path.dirname(os.path.abspath("__file__")) + "/"  # "E:/Workspace_WFU/ATN/Auto/"
    times = int(opt.num)
    create_empty_folders(main_path, opt.data)
    if len(opt.comment) > 2:
        comments = platform.platform() + ": " + opt.comment
    else:
        comments = platform.platform()

    data_x = load_data(main_path, "/data/data_x/data_x_{}.npy".format(opt.data))
    # print("raw_path:", "/data/data_x/data_x_{}{}.npy".format(opt.data, "" if ("alpha" in opt.data or "beta" in opt.data) else "_raw"))
    data_x_raw = load_data(main_path, "/data/data_x/data_x_{}{}.npy".format(opt.data, "" if "alpha" in opt.data else "_raw"))
    base_dic, base_res, base_res_inter = initial_record(main_path, opt, data_x_raw, opt.data, int(opt.kmeans), int(opt.k))
    start_index = get_start_index(main_path, opt)

    for i in range(times):
        j, p, ds = train(main_path, opt, data_x, start_index + i, params, base_dic, base_res, base_res_inter)
        save_record(main_path, opt, start_index + i, ds, j, p, comments, opt.data, params)
        # get_start_index(main_path, opt.data)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", default=500, help="number of training")
    parser.add_argument("--comment", default="", help="any comment displayed on record")
    parser.add_argument("--data", default="alpha1", help="dataset of data_x  ('$DATA_TYPE$ID', $DATA_TYPE could be 'Alpha', 'Gamma', 'Delta', 'Epsilon', ..., 'Iota'; $ID could be 1,2,3,4. e.g. 'delta1', 'eta2')")
    parser.add_argument("--kmeans", default=0, help="time of doing kmeans as base before training")
    parser.add_argument("--k", default=6, help="number of clusters needed")
    parser.add_argument("--clear", default=1, help="whether to clear the proposed file")
    parser.add_argument("--main_epoch", default=1000, help="iteration in main algorithm")
    parser.add_argument("--alpha", default=0.00001, help="alpha parameter in model (not dataset names)")
    parser.add_argument("--beta", default=0.1, help="beta parameter in model (not dataset names)")
    parser.add_argument("--h_dim", default=8, help="h_dim_FC & h_dim_RNN")
    opt = parser.parse_args()
    params = {
        # [Step 2] Define network parameters
        'K': int(opt.k),              # 5
        'h_dim_FC': int(opt.h_dim),   # 26
        'h_dim_RNN': int(opt.h_dim),  # 26
        'num_layer_encoder': 2,       # 2
        'num_layer_selector': 3,      # 3
        'num_layer_predictor': 2,     # 2
        # [Step 3] Initialize network
        'lr_rate': 0.0001,            # 0.0001
        'keep_prob_s3': 0.5,          # 0.5
        'mb_size_s3': 32,             # 32
        'iteration_s3': 1000,         # 3750
        'check_step_s3': 100,         # 250
        # [Step 4] Train temporal phenotyping
        'alpha': float(opt.alpha),    # 0.00001
        'beta': float(opt.beta),      # 1
        'mb_size_s4': 32,             # 8
        'keep_prob_s4': 0.7,          # 0.7
        'lr_rate1': 0.0001,           # 0.0001
        'lr_rate2': 0.0001,           # 0.0001
        # [Step 6] Initializing embedding & selector
        'iteration_s6': 15000,        # 15000
        'check_step_s6': 1000,        # 1000
        # [Step 7] Training main algorithm
        'iteration_s7': int(opt.main_epoch),       # 5000
        'check_step_s7': 100          # 100
    }
    print(json.dumps(params, indent=4, ensure_ascii=False))
    start(params, opt)






