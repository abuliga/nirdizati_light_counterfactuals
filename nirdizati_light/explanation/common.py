from enum import Enum

from nirdizati_light.explanation.wrappers.dice_wrapper import dice_explain
class ExplainerType(Enum):
    DICE = 'dice'



def explain(CONF, predictive_model, encoder, cf_df=None, test_df=None, df=None, query_instances=None,
            method=None, optimization=None, support=0.9, timestamp_col_name=None,
            model_path=None,case_ids=None,random_seed=None,adapted=None,filtering=None):
    explainer = CONF['explanator']
    if explainer is ExplainerType.DICE.value:
        return dice_explain(CONF, predictive_model, encoder=encoder, cf_df=cf_df, df=df, query_instances=query_instances,
                            method=method, optimization=optimization,
                            support=support, timestamp_col_name=timestamp_col_name,model_path=model_path,case_ids=case_ids,
                            random_seed=random_seed,adapted=adapted,filtering=filtering)
