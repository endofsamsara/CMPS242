import codecs


class VectorWriter:
    def __init__(self, filepath):
        self.out = codecs.open(filepath, 'w', 'utf-8')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_write()

    def write_row(self, label, vector):
        self.out.write(str(label) + ' ')
        for n in range(len(vector)):
            if vector[n] != 0:
                self.out.write(str(n) + ':' + str(vector[n]) + ' ')
        self.out.write('\n')

    def _end_write(self):
        self.out.flush()
        self.out.close()
