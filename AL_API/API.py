import json
import clr
import sys

idea_path = r"C:\Program Files\IDEA StatiCa\StatiCa 24.0"
# idea_path = r"c:\code\fem\ai\data_generation\containers\compute_node_core\app"
# idea_path = r"c:\code\fem\IdeaStatiCa\bin\Release\net48"
# idea_path = r"c:\code\fem\IdeaStatiCa\bin\Debug\net48"
assembly_path = idea_path

# modify path to be able to load .net assemblies
sys.path.append(assembly_path)

# load the assembly IdeaStatiCa.Plugin which is responsible for communication with IdeaStatiCa
clr.AddReference('IdeaStatiCa.Plugin')
from IdeaStatiCa.Plugin import ConnHiddenClientFactory

class IdeaConnectionClient:
    def __init__(self):
        self.idea_connection_client = None
        self.connection = None

    def open(self, filename):
        # create the instance of the client which communicates with IdeaStatiCa
        factory = ConnHiddenClientFactory(idea_path)
        self.idea_connection_client = factory.Create()

        print(filename)

        # open idea connection project
        self.idea_connection_client.OpenProject(filename)

        # get information about connections in the project and print them
        projectInfo = self.idea_connection_client.GetProjectInfo()

        self.connection = projectInfo.Connections[0]

    def get_params(self):
        params_json_string = self.idea_connection_client.GetParametersJSON(self.connection.Identifier)
        connection_params = json.loads(params_json_string)
        return connection_params

    def _apply_params(self, params: dict) -> bool:
        updated_params_json_string = json.dumps(params)
        success = self.idea_connection_client.ApplyParameters(self.connection.Identifier, updated_params_json_string)

        # string return value is to be updated in the future to boolean
        if success == "True":
            return True
        elif success == "False":
            return False
        else:
            raise Exception("Unexpected return value from ApplyParameters().")

    def _update_param_by_name(self, params, param_key, param_value):
        for i, v in enumerate(params):
            if params[i]['identifier'] == param_key:
                # params have to be set as strings
                # otherwise a weird conversion happens
                # where float 0.3 gets converted to string '0,3' (depending on environment)
                # and setting the expression '0,3' fails, because numbers have to have a dot separator...
                params[i]['value'] = str(param_value)
                return

        raise Exception('Parameter with the identifier \'' + str(param_key) + '\' not found')

    def update_params_by_name(self, new_params: dict) -> bool:
        """
        Sets new param values.

        :param new_params: dict where key is the name of the parameter and value is the desired value
        :returns True on success, False otherwise (if params couldn't be set or set to model)
        """
        params = self.get_params()

        for key, value in new_params.items():
            self._update_param_by_name(params, key, value)

        success = self._apply_params(params)

        return success

    def params_valid(self):
        """
        Checks whether none of the required validations for the parameters have failed.
        """
        params = self.get_params()
        for param in params:
            value = param['evaluatedValidationValue']

            match value:
                case None:
                    pass
                case 'True':
                    pass
                case 'Error':
                    return False
                case 'False':
                    return False
                case _:
                    raise Exception("Unexpected value of evaluatedValidationValue: " + str(value))

        return True


    def get_param_evaluated_value_by_name(self, name):
        params = self.get_params()
        for i, v in enumerate(params):
            if params[i]['identifier'] == name:
                return params[i]['evaluatedValue']

        raise Exception('Parameter with the identifier \'' + str(name) + '\' not found')


    def get_loads(self):
        loading_json = self.idea_connection_client.GetConnectionLoadingJSON(self.connection.Identifier)
        loads = json.loads(loading_json)
        return loads

    def set_loads(self, loads):
        loads_json = json.dumps(loads)
        result = self.idea_connection_client.UpdateLoadingFromJson(self.connection.Identifier, loads_json)

        if result != 'OK':
            raise Exception('UpdateLoadingFromJson didn''t return ''OK''')

    def save_as(self, filename):
        self.idea_connection_client.SaveAsProject(filename)

    def calculate(self) -> bool:
        brief_results = self.idea_connection_client.Calculate(self.connection.Identifier)

        if not brief_results:
            return False

        # Calculation results don't contain any result summary. Calculation probably did not run.
        if len(brief_results.ConnectionCheckRes[0].CheckResSummary) == 0:
            return False

        return True

    def is_overloaded(self):
        brief_results = self.idea_connection_client.Calculate(self.connection.Identifier)

        if not brief_results:
            return False

        if len(brief_results.ConnectionCheckRes) == 0:
            return False

        if not brief_results.ConnectionCheckRes[0].Messages:
            return False

        nr_messages = len(brief_results.ConnectionCheckRes[0].Messages.Messages)

        if nr_messages == 0:
            return False

        for i in range(nr_messages):
            message = brief_results.ConnectionCheckRes[0].Messages.Messages[i].Message
            if "overloaded" in message:
                return True

        return False


    def get_results(self):
        check_results_json = self.idea_connection_client.GetCheckResultsJSON(self.connection.Identifier)
        check_results = json.loads(check_results_json)
        return check_results

    def close(self):
        self.idea_connection_client.Close()
