from ml_socket import *


class FL_Server():
    def __init__(self, curr_model, num_clients, rounds):
        self.s = socket.socket()
        self.executor = concurrent.futures.ThreadPoolExecutor(num_clients)
        self.curr_model = curr_model
        self.num_clients = num_clients
        self.rounds = rounds
        self.results = []
        self.connections = []

    def __handle_client(self, c, model, send_rounds=False):
        if send_rounds:
            c.send(str(self.rounds).encode(FORMAT))
        ml_send(c, model)
        new_w = ml_recv(c, SIZE)
        return new_w

    def __initiate_socket(self):
        self.s.bind(('', PORT))
        print("\n[BINDED] socket binded to %s.\n" % (PORT))

        self.s.listen(self.num_clients)
        print("\n[LISTENING] socket is listening.\n")

    def __accept_connection(self):
        c, addr = self.s.accept()
        self.connections.append(c)
        print("")
        print('[NEW CONNECTION] Got connection from', addr)
        res = self.executor.submit(
            self.__handle_client, c, copy.deepcopy(self.curr_model), True)
        self.results.append(res)

    def __fl_policy(self, new_weights):
        avg_weights = []
        for layer in range(len(new_weights[0])):
            sum_layer = np.zeros_like(new_weights[0][layer])
            for new_weight in new_weights:
                sum_layer += new_weight[layer]
            avg_weights.append(sum_layer/self.num_clients)
        self.curr_model.set_weights(avg_weights)

    def __fl_loop(self):
        for i in range(self.rounds):
            new_weights = []
            for f in concurrent.futures.as_completed(self.results):
                model = f.result()
                new_weights.append(model.get_weights())

            self.__fl_policy(new_weights)

            new_weights = []
            self.results = []

            if i != self.rounds - 1:
                for c in self.connections:
                    res = self.executor.submit(
                        self.__handle_client, c, copy.deepcopy(self.curr_model))
                    self.results.append(res)

            print(f'\n✅ ROUND {i+1} COMPLETED.\n')

    def __close_connections(self):
        for c in self.connections:
            c.close()
        print("[CONNECTIONS CLOSED]")
        print("")

    def train(self):
        os.system(f'rm -rf ./pyfl_logs/')
        self.__initiate_socket()
        for _ in range(self.num_clients):
            self.__accept_connection()
            if len(self.connections) == self.num_clients:
                self.__fl_loop()
                self.__close_connections()
        self.s.close()
        os.system(f'tensorboard --logdir={PATH}')

    def test(self, data, target, loss, optimizer, lr, metrics):
        print("\n\n🔍 Testing...\n\n")
        self.curr_model.compile(loss=loss,
                                optimizer=optimizer(lr),
                                metrics=metrics)
        self.curr_model.evaluate(data, target)


class FL_Client():
    def __init__(self, name, data, target):
        self.name = name
        self.data = data
        self.target = target
        self.s = socket.socket()

    def train(self, epochs, batch_size, lr, loss, optimizer, metrics):
        self.s.connect((IP, PORT))
        print(f"[NEW CONNECTION] to {IP}:{PORT}")

        rounds = int(self.s.recv(SIZE).decode(FORMAT))
        for i in range(rounds):
            log_dir = f"{tensorboard_path}/{self.name}/round_{i+1}/" + \
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, update_freq="batch")
            model = ml_recv(self.s, SIZE)
            model.compile(loss=loss,
                          optimizer=optimizer(lr),
                          metrics=metrics)
            model.fit(self.data,
                      self.target,
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=[tensorboard_callback])
            ml_send(self.s, model)
            print(f'\n🛎️ ROUND {i+1} COMPLETED.\n')

        self.s.close()