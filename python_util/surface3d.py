import numpy as np

from bokeh.core.properties import Instance, String, Float
from bokeh.io import show
from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.util.compiler import TypeScript

TS_CODE = """
// This custom model wraps one part of the third-party vis.js library:
//
//     http://visjs.org/index.html
//
// Making it easy to hook up python data analytics tools (NumPy, SciPy,
// Pandas, etc.) to web presentations using the Bokeh server.

import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"
import {ColumnDataSource} from "models/sources/column_data_source"
import {LayoutItem} from "core/layout"
import * as p from "core/properties"

declare namespace vis {
  class Graph3d {
    constructor(el: HTMLElement, data: object, OPTIONS: object)
    setData(data: vis.DataSet): void
  }

  class DataSet {
    add(data: unknown): void
  }
}

// This defines some default options for the Graph3d feature of vis.js
// See: http://visjs.org/graph3d_examples.html for more details.
const OPTIONS = {
  width: '600px',
  height: '600px',
  style: 'dot-color',
  xMax: 30,
  xMin: 0,
  yMax: 30,
  yMin: 0,
  zMax: 2,
  zMin: -1,
  xLabel: 'juv_fel_count',
  yLabel: 'priors_count',
  zLabel: 'Race_African_American',
  showPerspective: true,
  showGrid: true,
  keepAspectRatio: true,
  verticalRatio: 1.0,
  legendLabel: 'viability',
  cameraPosition: {
    horizontal: -0.35,
    vertical: 0.22,
    distance: 1.8,
  },
}
// To create custom model extensions that will render on to the HTML canvas
// or into the DOM, we must create a View subclass for the model.
//
// In this case we will subclass from the existing BokehJS ``LayoutDOMView``
export class Surface3dView extends LayoutDOMView {
  model: Surface3d

  private _graph: vis.Graph3d

  initialize(): void {
    super.initialize()

    const url = "https://cdnjs.cloudflare.com/ajax/libs/vis/4.16.1/vis.min.js"
    const script = document.createElement("script")
    script.onload = () => this._init()
    script.async = false
    script.src = url
    document.head.appendChild(script)
  }

  private _init(): void {
    // Create a new Graph3s using the vis.js API. This assumes the vis.js has
    // already been loaded (e.g. in a custom app template). In the future Bokeh
    // models will be able to specify and load external scripts automatically.
    //
    // BokehJS Views create <div> elements by default, accessible as this.el.
    // Many Bokeh views ignore this default <div>, and instead do things like
    // draw to the HTML canvas. In this case though, we use the <div> to attach
    // a Graph3d to the DOM.
    this._graph = new vis.Graph3d(this.el, this.get_data(), OPTIONS)

    // Set a listener so that when the Bokeh data source has a change
    // event, we can process the new data
    this.connect(this.model.data_source.change, () => {
      this._graph.setData(this.get_data())
    })
  }

  // This is the callback executed when the Bokeh data has an change. Its basic
  // function is to adapt the Bokeh data source to the vis.js DataSet format.
  get_data(): vis.DataSet {
    const data = new vis.DataSet()
    const source = this.model.data_source
    for (let i = 0; i < source.get_length()!; i++) {
      data.add({
        x: source.data[this.model.x_good][i],
        y: source.data[this.model.y_good][i],
        z: source.data[this.model.z_good][i],
        style:  0
      });
      data.add({
        x: source.data[this.model.x_bad][i],
        y: source.data[this.model.y_bad][i],
        z: source.data[this.model.z_bad][i],
        style:  10
      });
      data.add({
        x: source.data[this.model.x_okay][i],
        y: source.data[this.model.y_okay][i],
        z: source.data[this.model.z_okay][i],
        style:  5
      });
      data.add({
        x: source.data[this.model.x_best][i],
        y: source.data[this.model.y_best][i],
        z: source.data[this.model.z_best][i],
        style:  3
      });
      data.add({
        x: this.model.x_indiv,
        y: this.model.y_indiv,
        z: this.model.z_indiv,
        style:  {fill: 'black'}
      });
    }
    return data
  }

  get child_models(): LayoutDOM[] {
    return []
  }

  _update_layout(): void {
    this.layout = new LayoutItem()
    this.layout.set_sizing(this.box_sizing())
  }
}

// We must also create a corresponding JavaScript BokehJS model subclass to
// correspond to the python Bokeh model subclass. In this case, since we want
// an element that can position itself in the DOM according to a Bokeh layout,
// we subclass from ``LayoutDOM``
export namespace Surface3d {
  export type Attrs = p.AttrsOf<Props>

  export type Props = LayoutDOM.Props & {
    x_good: p.Property<string>
    y_good: p.Property<string>
    z_good: p.Property<string>
    
    x_bad: p.Property<string>
    y_bad: p.Property<string>
    z_bad: p.Property<string>
    
    x_best: p.Property<string>
    y_best: p.Property<string>
    z_best: p.Property<string>
    
    x_okay: p.Property<string>
    y_okay: p.Property<string>
    z_okay: p.Property<string>
    
    x_indiv: p.Property<number>
    y_indiv: p.Property<number>
    z_indiv: p.Property<number>
        
    data_source: p.Property<ColumnDataSource>
  }
}

export interface Surface3d extends Surface3d.Attrs {}

export class Surface3d extends LayoutDOM {
  properties: Surface3d.Props
  __view_type__: Surface3dView

  constructor(attrs?: Partial<Surface3d.Attrs>) {
    super(attrs)
  }

  // The ``__name__`` class attribute should generally match exactly the name
  // of the corresponding Python class. Note that if using TypeScript, this
  // will be automatically filled in during compilation, so except in some
  // special cases, this shouldn't be generally included manually, to avoid
  // typos, which would prohibit serialization/deserialization of this model.
  static __name__ = "Surface3d"

  static init_Surface3d() {
    // This is usually boilerplate. In some cases there may not be a view.
    this.prototype.default_view = Surface3dView

    // The @define block adds corresponding "properties" to the JS model. These
    // should basically line up 1-1 with the Python model class. Most property
    // types have counterparts, e.g. ``bokeh.core.properties.String`` will be
    // ``p.String`` in the JS implementatin. Where the JS type system is not yet
    // as rich, you can use ``p.Any`` as a "wildcard" property type.
    this.define<Surface3d.Props>({
      x_good:            [ p.String   ],
      y_good:            [ p.String   ],
      z_good:            [ p.String   ],
      
      x_bad:            [ p.String   ],
      y_bad:            [ p.String   ],
      z_bad:            [ p.String   ],
      
      x_okay:            [ p.String   ],
      y_okay:            [ p.String   ],
      z_okay:            [ p.String   ],
      
      x_best:            [ p.String   ],
      y_best:            [ p.String   ],
      z_best:            [ p.String   ],
      
      x_indiv:            [ p.Number   ],
      y_indiv:            [ p.Number   ],
      z_indiv:            [ p.Number   ],
      
      data_source:  [ p.Instance ],
    })
  }
}
"""

# This custom extension model will have a DOM view that should layout-able in
# Bokeh layouts, so use ``LayoutDOM`` as the base class. If you wanted to create
# a custom tool, you could inherit from ``Tool``, or from ``Glyph`` if you
# wanted to create a custom glyph, etc.
class Surface3d(LayoutDOM):

    # The special class attribute ``__implementation__`` should contain a string
    # of JavaScript code that implements the browser side of the extension model.
    __implementation__ = TypeScript(TS_CODE)

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://docs.bokeh.org/en/latest/docs/reference/core/properties.html#bokeh-core-properties

    # This is a Bokeh ColumnDataSource that can be updated in the Bokeh
    # server by Python code
    data_source = Instance(ColumnDataSource)

    # The vis.js library that we are wrapping expects data for x, y, and z.
    # The data will actually be stored in the ColumnDataSource, but these
    # properties let us specify the *name* of the column that should be
    # used for each field.
    x_good = String
    y_good = String
    z_good = String
    
    x_bad = String
    y_bad = String
    z_bad = String
    
    x_best = String
    y_best = String
    z_best = String
    
    x_okay = String
    y_okay = String
    z_okay = String
    
    x_indiv = Float
    y_indiv = Float
    z_indiv = Float
    