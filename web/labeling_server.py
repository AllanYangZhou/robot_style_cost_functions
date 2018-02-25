from flask import (
    Flask, render_template,
    send_from_directory, request,
    redirect, url_for
)
from mongoengine import connect
from ComparisonDocument import Comparison
import os
app = Flask(__name__)

connect('style_experiment')

@app.route('/')
def index():
    unlabeled = Comparison.objects(label=None)
    if len(unlabeled):
        c = unlabeled[0]
        context = {
            'pathA': os.path.basename(c.pathA),
            'pathB': os.path.basename(c.pathB),
            'cid': c.id,
            'num_unlabeled': len(unlabeled)
        }
        return render_template(
            'compare.html', **context)
    else:
        return render_template('empty.html')


@app.route('/vids/<path:path>')
def send_vid(path):
    return send_from_directory('vids', path)


@app.route('/submit', methods=['POST'])
def handle_submit():
    cid = request.form['cid']
    label = request.form['label']
    c = Comparison.objects(id=cid)[0]
    if label == 'A':
        c.label = 0
    elif label == 'B':
        c.label = 1
    elif label == 'Neither':
        c.label = -1
    else:
        raise Exception('Unexpected label.')
    c.save()
    return redirect(url_for('index'))