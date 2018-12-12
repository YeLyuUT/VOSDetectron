from easydict import EasyDict

class vos_imdb():
  def __init__(self,name):
    self._edicts = []
    self._dbname = name
    self._sequences = []

  def add_seq_info_dict(self):
    self._edicts.append(EasyDict())
    #return dict and its index
    return self._edicts[-1],len(self._edicts)-1

  def get_seq_info(self, ind):
    return self._edicts[ind]

  def num_sequence(self):
    return len(self._sequences)

  def get_sequence(self,index):
    return self._sequences[index]

  def get_image(self,seq_ind,t_ind):
    """
    Args:
      seq_ind: seqence index.
      t_ind: time index in a sequence.
    """
    pass

  def get_label(self,seq_ind,t_ind):
    """
    Args:
      seq_ind: seqence index.
      t_ind: time index in a sequence.
    """
    pass
  
  @property
  def name(self):
    return self._dbname
