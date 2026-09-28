# ABOUTME: Run the official grader for local Inspect fixtures without inspecting every unrelated cached image.
# ABOUTME: Only the tag inventory query is optimized; patch application, tests and scoring are unchanged.
import runpy
from docker.models.images import ImageCollection, Image


def sparse_list(self, name=None, all=False, filters=None):
    return [Image(attrs=item,client=self.client,collection=self)
            for item in self.client.api.images(name=name,all=all,filters=filters)]


if __name__ == '__main__':
    ImageCollection.list = sparse_list
    runpy.run_module('swebench.harness.run_evaluation', run_name='__main__')
