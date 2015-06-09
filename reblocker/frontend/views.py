import json
import os

from flask import (abort, Blueprint, current_app, flash, g, redirect,
                    render_template, request, url_for)

from flask.ext.classy import FlaskView, route

import fiona
from fiona.crs import to_string
from werkzeug import secure_filename

from ..reblocker import reblocked_JSON

frontend = Blueprint('frontend', __name__, url_prefix='')

class MainView(FlaskView):
    ''' Our base View
    '''
    route_base = '/'

    @route('/', endpoint='index')
    def index(self):
        ''' Our main index view '''
        return render_template('frontend/index.html')

    @route('/', endpoint='upload', methods=['POST'])
    def upload(self):
        uploaded_files = request.files.getlist("file[]")
        shapefile = None

        for file in uploaded_files:
            if file: # and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                if filename.rsplit('.', 1)[1] == 'shp':
                    shapefile = filepath
        if shapefile:
            # Read shapefile, throw error if we don't have everythign we need
            with fiona.open(shapefile) as c:
                    geoShapes = []
                    proj = to_string(c.crs)
                    for i, item in c.items():
                        geoJSON = {
                            'geometry': {
                                'type' : item['geometry']['type'],
                                'coordinates' : [[list(tup) for tup in coord] for coord in item['geometry']['coordinates']],
                            },
                            'type': item['type'],
                            #'features': json.dumps(item['properties']),
                            'crs' : {
                                'type' : 'name',
                                'properties' : {
                                    'name':"urn:ogc:def:crs:EPSG::3857"
                                }
                            },
                        }
                        geoShapes.append(geoJSON)

            reblocked_geoShapes = reblocked_JSON(shapefile)

            flash('File(s) uploaded successfully')
            return render_template('frontend/index.html',
                                    shapes=geoShapes, proj=proj, roads=reblocked_geoShapes)

        flash('Please upload a shapefile')
        return render_template('frontend/index.html')



#Register our View Class
MainView.register(frontend)
