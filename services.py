import operator
import ast
import itertools


class KerasGridSearchService:

    @staticmethod
    def keras_grid_search(k_model, param_grid, x_train, y_train, x_validation, y_validation, keras_backend):
        """
        For a given model and parameter grid, trains on a train datasets and evaluates with a validation dataset
        :param k_model:
        :param param_grid:
        :param x_train:
        :param y_train:
        :param x_validation:
        :param y_validation:
        :param keras_backend:
        :return:
        """
        grid_res = KerasGridSearchService.build_grid(param_grid)
        best_search = {}
        for gr in grid_res:
            model = k_model(input_dim=len(x_train[0]), output_dim=len(y_train[0]),
                            **{x: gr[x] for x in gr})
            model.fit(x_train, y_train, epochs=gr['epochs'], batch_size=gr['batch_size'], verbose=0, shuffle=False)
            # Final evaluation of the model
            scores = model.evaluate(x_validation, y_validation, verbose=0)
            best_search.setdefault(str(gr), 0)
            best_search[str(gr)] = scores[1]
            print("Parameters: " + str(gr) + " Accuracy: %.2f%%" % (scores[1] * 100))
            if keras_backend.backend() == 'tensorflow':
                print('Clearing backend session')
                keras_backend.clear_session()

        return best_search

    @staticmethod
    def build_grid(search_space: dict):
        """
        Gives all possible combinations for a search space
        :param search_space:
        :return:
        """
        keys, values = zip(*search_space.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    @staticmethod
    def select_top_scores(scores_dict, percent=0.2):
        """
        Selects the top percent(default = 20% [0-1]) values of a given dictionary of key value pairs

        :param scores_dict:
        :param percent:
        :return:
        """
        scores_dict = sorted(scores_dict.items(), key=operator.itemgetter(1), reverse=True)

        return dict(scores_dict[:int(len(scores_dict) * percent)])

    @staticmethod
    def scores_to_param_dict(scores_dict):
        """
        Converts the scores_dictionary ( result from select_top_scores(_) )
        to param_dict to be used in keras_grid_search(_)
        :param scores_dict:
        :return:
        """
        param_dict = {}
        for sa in scores_dict:
            curr_dict = ast.literal_eval(sa)
            for param in curr_dict:
                param_dict.setdefault(param, {})
                param_dict[param].setdefault(curr_dict[param], [])
                if curr_dict[param] not in param_dict[param]:
                    param_dict[param].append(curr_dict[param])
        return param_dict
