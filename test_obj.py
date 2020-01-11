import generate_templates
import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name = 'run1', checkpoint_dir='checkpoint')




answer = generate_scenario(n_objects,n_containers,obj_type)

